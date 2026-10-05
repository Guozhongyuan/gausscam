"""wgpu 4DGS render pipeline (P2-promoted from examples/wgpu/w3_sort_opt.py).

OneSweep ("opt", NVIDIA) and hierarchical-colscan ("optcol", everything else)
radix variants over the _wgpu_base WGSL; Pipeline renders C cameras of a
merged scene+robot Gaussian cloud. K comes from make_K(fovy) and can be
re-targeted by overriding d["W"/"H"/"fovy"] before construction; set_cam()
re-uploads viewmats/cull-origin per frame (moving-camera support).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import wgpu
from wgpu.backends.wgpu_native.extras import write_timestamp

from gausscam.backends import _wgpu_base as w2
from gausscam.backends._wgpu_base import make_K  # noqa: F401 -- re-export


TILE = w2.TILE
CAP = w2.CAP                       # historical default capacity (4M pairs)
CAP_MAX = 64_000_000               # capacity ceiling: ~1.8 GB of sort scratch
CHUNK = 1024                       # opt sweep block: 256 threads x 4 items
NPASS = w2.NPASS
T_CHUNKS = 32                      # colscan tile: 32 chunks x 256 bins


def auto_capacity(n: int) -> int:
    """Scene-sized (gaussian, tile) pair capacity for Pipeline(cap=None).

    Every visible splat covers >= 1 tile, so pairs grow with the in-view
    count; 4x N covers interior scenes with headroom (~2 pairs/gaussian
    measured on a 7.7M church scan). Small scenes keep the historical 4M
    footprint; the ceiling bounds pathological clouds.
    """
    return int(min(CAP_MAX, max(w2.CAP, 4 * int(n))))

MARKER_BEFORE = {"rank_0": 1, "sweep_0": 1, "tile_starts": 2}


# ---------------------------------------------------------------- WGSL ----- #

def opt_radix_kernels(p: int, chunks_opt: int) -> str:
    """L4 OneSweep pass (CUB/Orochi-style, clean-room WGSL): count -> bin_scan
    -> sweep. The sweep publishes per-block AGGREGATE digit counts, resolves
    its cross-block exclusive base per bin via decoupled look-back (spin on a
    2-state word), ranks elements with the order-contiguous group trick, and
    scatters — all in one dispatch. No hist2d matrix, no colscan chain.
    Status arrays ping-pong by pass parity; both are re-zeroed each frame.
    chunks_opt: sweep blocks per pass, from the Pipeline's pair capacity."""
    word_sel = "1u" if p < 4 else "0u"
    shift = 8 * (p % 4)
    src = "a" if p % 2 == 0 else "b"
    dst = "b" if p % 2 == 0 else "a"
    decl = ""
    if p == 0:
        decl = (
            f"@group(0) @binding(34) var<storage, read_write> sw_status: "
            f"array<atomic<u32>>;   // [{NPASS}*{chunks_opt}*256] lookback words\n"
            f"@group(0) @binding(35) var<storage, read_write> bin_cnt: "
            f"array<atomic<u32>>;   // [{NPASS}*256]\n"
            "\n// frame-start zero of all status planes + pass counters\n"
            "@compute @workgroup_size(256)\n"
            "fn zero_sweep(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
            f"    let stat_words = {NPASS * chunks_opt * 256}u;\n"
            f"    let total = {NPASS * chunks_opt * 256 + NPASS * 256}u;\n"
            "    // grid-stride: the status planes can outgrow one dispatch\n"
            "    for (var i = gid.x; i < total; i += 16776960u) {\n"
            "        if (i < stat_words) { atomicStore(&sw_status[i], 0u); }\n"
            "        else { atomicStore(&bin_cnt[i - stat_words], 0u); }\n"
            "    }\n"
            "}\n")
    return decl + f"""
fn digit_{p}(i: u32) -> u32 {{
    let word = keys_{src}[i*2u + {word_sel}];
    return (word >> {shift}u) & 0xffu;
}}

var<workgroup> sw_dig_{p}: array<u32, {CHUNK}>;         // digits, element order
var<workgroup> sw_histg_{p}: array<atomic<u32>, 2048>;  // 8 groups x 256 bins
var<workgroup> sw_cntg_{p}: array<u32, 2048>;           // per-group running ranks
var<workgroup> sw_lk_{p}: array<u32, 256>;              // lookback block bases
var<workgroup> sw_rank_{p}: array<u32, {CHUNK}>;        // in-block ranks
var<workgroup> sw_h_{p}: array<atomic<u32>, 256>;       // count-kernel hist

// upsweep: per-block digit histogram -> global per-pass bin counters
@compute @workgroup_size(256)
fn count_{p}(@builtin(workgroup_id) wgid: vec3<u32>,
             @builtin(local_invocation_id) lid: vec3<u32>) {{
    let c = wgid.x;
    let n = atomicLoad(&counts[P.c * P.tile_w * P.tile_h]);
    let start = c * {CHUNK}u;
    if (start >= n) {{ return; }}
    atomicStore(&sw_h_{p}[lid.x], 0u);
    workgroupBarrier();   // zero visible before any digit add
    for (var k = 0u; k < 4u; k++) {{
        let i = start + lid.x + k*256u;
        if (i < n) {{ atomicAdd(&sw_h_{p}[digit_{p}(i)], 1u); }}
    }}
    workgroupBarrier();
    atomicAdd(&bin_cnt[{p} * 256u + lid.x], sw_h_{p}[lid.x]);
}}

// exclusive scan of this pass's 256 bin counters -> bin bases
var<workgroup> bs_partial_{p}: array<u32, 256>;

@compute @workgroup_size(256)
fn bin_scan_{p}(@builtin(local_invocation_id) lid: vec3<u32>) {{
    let tid = lid.x;
    var v = atomicLoad(&bin_cnt[{p} * 256u + tid]);
    bs_partial_{p}[tid] = v;
    workgroupBarrier();
    var off = 1u;
    loop {{
        if (off >= 256u) {{ break; }}
        if (tid >= off) {{ v = v + bs_partial_{p}[tid - off]; }}
        workgroupBarrier();
        if (tid >= off) {{ bs_partial_{p}[tid] = v; }}
        workgroupBarrier();
        off = off << 1u;
    }}
    bin_base[tid] = v - atomicLoad(&bin_cnt[{p} * 256u + tid]);
}}

// the sweep: group ranks + decoupled look-back + scatter in one dispatch
@compute @workgroup_size(256)
fn sweep_{p}(@builtin(workgroup_id) wgid: vec3<u32>,
             @builtin(local_invocation_id) lid: vec3<u32>) {{
    let c = wgid.x;
    let n = atomicLoad(&counts[P.c * P.tile_w * P.tile_h]);
    let start = c * {CHUNK}u;
    if (start >= n) {{ return; }}
    let par = {p}u;
    for (var z = 0u; z < 8u; z++) {{
        atomicStore(&sw_histg_{p}[z*256u + lid.x], 0u);
        sw_cntg_{p}[z*256u + lid.x] = 0u;
    }}
    var d0 = 0xffffffffu;
    var d1 = 0xffffffffu;
    var d2 = 0xffffffffu;
    var d3 = 0xffffffffu;
    let i0 = start + lid.x;
    if (i0 < n) {{ d0 = digit_{p}(i0); }}
    sw_dig_{p}[lid.x] = d0;
    let i1 = i0 + 256u;
    if (i1 < n) {{ d1 = digit_{p}(i1); }}
    sw_dig_{p}[lid.x + 256u] = d1;
    let i2 = i0 + 512u;
    if (i2 < n) {{ d2 = digit_{p}(i2); }}
    sw_dig_{p}[lid.x + 512u] = d2;
    let i3 = i0 + 768u;
    if (i3 < n) {{ d3 = digit_{p}(i3); }}
    sw_dig_{p}[lid.x + 768u] = d3;
    workgroupBarrier();
    if (d0 != 0xffffffffu) {{ atomicAdd(&sw_histg_{p}[(lid.x / 128u)*256u + d0], 1u); }}
    if (d1 != 0xffffffffu) {{ atomicAdd(&sw_histg_{p}[((lid.x + 256u) / 128u)*256u + d1], 1u); }}
    if (d2 != 0xffffffffu) {{ atomicAdd(&sw_histg_{p}[((lid.x + 512u) / 128u)*256u + d2], 1u); }}
    if (d3 != 0xffffffffu) {{ atomicAdd(&sw_histg_{p}[((lid.x + 768u) / 128u)*256u + d3], 1u); }}
    workgroupBarrier();
    // cross-group exclusive scan; running = block total for bin lid.x
    var total = 0u;
    for (var g = 0u; g < 8u; g++) {{
        let v = atomicLoad(&sw_histg_{p}[g*256u + lid.x]);
        atomicStore(&sw_histg_{p}[g*256u + lid.x], total);
        total = total + v;
    }}
    // publish AGGREGATE (state 1) for this block/bin
    atomicExchange(&sw_status[par*{chunks_opt}u*256u + c*256u + lid.x],
                   (1u << 30u) | total);
    // decoupled look-back: exclusive sum over predecessor blocks, bin lid.x
    var base = 0u;
    var j = c;
    loop {{
        if (j == 0u) {{ break; }}
        j = j - 1u;
        var s = atomicLoad(&sw_status[par*{chunks_opt}u*256u + j*256u + lid.x]);
        loop {{
            if ((s >> 30u) != 0u) {{ break; }}
            s = atomicLoad(&sw_status[par*{chunks_opt}u*256u + j*256u + lid.x]);
        }}
        base = base + (s & 0x3fffffffu);
        if ((s >> 30u) == 2u) {{ break; }}   // PREFIX ends the walk
    }}
    atomicExchange(&sw_status[par*{chunks_opt}u*256u + c*256u + lid.x],
                   (2u << 30u) | (base + total));
    sw_lk_{p}[lid.x] = base;
    workgroupBarrier();
    // ordered ranks: one leader per 128-element group, element order
    if (lid.x % 32u == 0u) {{
        let g = lid.x / 32u;
        for (var jj = 0u; jj < 128u; jj++) {{
            let dd = sw_dig_{p}[g*128u + jj];
            if (dd != 0xffffffffu) {{
                sw_rank_{p}[g*128u + jj] = atomicLoad(&sw_histg_{p}[g*256u + dd])
                                          + sw_cntg_{p}[g*256u + dd];
                sw_cntg_{p}[g*256u + dd] = sw_cntg_{p}[g*256u + dd] + 1u;
            }}
        }}
    }}
    workgroupBarrier();
    // scatter: pos = global bin base + block base + in-block rank
    for (var k = 0u; k < 4u; k++) {{
        let il = lid.x + k*256u;
        let d = sw_dig_{p}[il];
        if (d != 0xffffffffu) {{
            let pos = bin_base[d] + sw_lk_{p}[d] + sw_rank_{p}[il];
            keys_{dst}[pos*2u] = keys_{src}[(start + il)*2u];
            keys_{dst}[pos*2u+1u] = keys_{src}[(start + il)*2u+1u];
            flat_{dst}[pos] = flat_{src}[start + il];
        }}
    }}
}}
"""


FUSED_RASTERIZE = """
// ---- rasterize + quantize fused: blend loop writes packed u8 rgb and
// half-even u16 depth directly (bitwise-identical ops to the W1 quantize
// kernel; skips the f32 renders round-trip). Validated in W1.
var<workgroup> ws_g: array<i32, 256>;
var<workgroup> ws_xy: array<vec3<f32>, 256>;
var<workgroup> ws_conic: array<vec3<f32>, 256>;

@compute @workgroup_size(16, 16)
fn rasterize(@builtin(workgroup_id) wgid: vec3<u32>,
             @builtin(local_invocation_id) lid: vec3<u32>) {
    let n_tiles = P.tile_w * P.tile_h;
    let image_id = wgid.x / n_tiles;
    let tile_id = wgid.x % n_tiles;
    let i = (tile_id / P.tile_w) * 16u + lid.y;
    let j = (tile_id % P.tile_w) * 16u + lid.x;
    let pix_id = (image_id * P.h + i) * P.w + j;
    let px = f32(j) + 0.5;
    let py = f32(i) + 0.5;
    let inside = i < P.h && j < P.w;
    var done = !inside;

    let sbase = image_id * n_tiles + tile_id;
    let range_start = starts[sbase];
    let range_end = starts[sbase + 1u];

    let tr = lid.y * 16u + lid.x;
    var T = 1.0;
    var pix_out = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    let cnt = max(0i, range_end - range_start);
    let n_batches = (u32(cnt) + 255u) / 256u;
    for (var b = 0u; b < n_batches; b++) {
        let batch_start = range_start + i32(256u * b);
        let idx = batch_start + i32(tr);
        if (idx < range_end) {
            let g = u32(flat_a[idx]);
            ws_g[tr] = i32(g);
            ws_xy[tr] = vec3<f32>(means2d[g*2u], means2d[g*2u+1u], opacity_b[g]);
            ws_conic[tr] = vec3<f32>(conics[g*3u], conics[g*3u+1u], conics[g*3u+2u]);
        }
        workgroupBarrier();
        let batch_size = min(256u, u32(max(0i, range_end - batch_start)));
        var t = 0u;
        loop {
            if (t >= batch_size) { break; }
            if (!done) {
                let conic = ws_conic[t];
                let xy_opac = ws_xy[t];
                let dx = xy_opac.x - px;
                let dy = xy_opac.y - py;
                let sigma = 0.5 * (conic.x * dx * dx + conic.z * dy * dy)
                            + conic.y * dx * dy;
                let alpha = min(0.999, xy_opac.z * exp(-sigma));
                if (!(sigma < 0.0) && alpha >= ALPHA_THRESHOLD) {
                    let next_T = T * (1.0 - alpha);
                    if (next_T <= 1e-4) {
                        done = true;
                    } else {
                        let g = u32(ws_g[t]);
                        let vis = alpha * T;
                        pix_out += vec4<f32>(colors[g*4u], colors[g*4u+1u],
                                             colors[g*4u+2u], colors[g*4u+3u]) * vis;
                        T = next_T;
                    }
                }
            }
            t = t + 1u;
        }
        workgroupBarrier();
    }

    if (inside) {
        let ru = u32(clamp(pix_out.x, 0.0, 1.0) * 255.0);
        let gu = u32(clamp(pix_out.y, 0.0, 1.0) * 255.0);
        let bu = u32(clamp(pix_out.z, 0.0, 1.0) * 255.0);
        atomicStore(&rgb_out[pix_id], (bu << 16u) | (gu << 8u) | ru);
        let dv = clamp(pix_out.w, 0.0, 65.535) * 1000.0;
        let vv = floor(dv);
        let fr = dv - vv;
        let res = select(vv + 1.0, vv,
                         (fr < 0.5) || (fr == 0.5 && (u32(vv) & 1u) == 0u));
        atomicStore(&depth_out[pix_id], u32(res));
    }
}
"""


def optcol_radix_kernels(p: int, maxb2: int) -> str:
    """Arc/Mesa-safe fallback: group-rank + hierarchical colscan (no spin
    loops — look-back deadlocks the ARL/Mesa combination even at 2 blocks).
    Same CHUNK=1024 geometry and fused structure as the OneSweep variant.
    maxb2: colscan block-count ceiling, from the Pipeline's capacity."""
    word_sel = "1u" if p < 4 else "0u"
    shift = 8 * (p % 4)
    src = "a" if p % 2 == 0 else "b"
    dst = "b" if p % 2 == 0 else "a"
    decl = ""
    if p == 0:
        decl = (
            f"@group(0) @binding(33) "
            f"var<storage, read_write> partials: array<atomic<u32>>;"
            f"  // [{maxb2}*256] colscan tile partials\n")
    return decl + f"""
fn digit_{p}(i: u32) -> u32 {{
    let word = keys_{src}[i*2u + {word_sel}];
    return (word >> {shift}u) & 0xffu;
}}

var<workgroup> rk_digits_{p}: array<u32, {CHUNK}>;
var<workgroup> rk_histg_{p}: array<atomic<u32>, 2048>;  // 8 groups x 256 bins
var<workgroup> rk_cntg_{p}: array<u32, 2048>;

@compute @workgroup_size(256)
fn rank_{p}(@builtin(workgroup_id) wgid: vec3<u32>,
            @builtin(local_invocation_id) lid: vec3<u32>) {{
    let c = wgid.x;
    let n = atomicLoad(&counts[P.c * P.tile_w * P.tile_h]);
    let start = c * {CHUNK}u;
    if (start >= n) {{ return; }}
    for (var z = 0u; z < 8u; z++) {{
        atomicStore(&rk_histg_{p}[z*256u + lid.x], 0u);
        rk_cntg_{p}[z*256u + lid.x] = 0u;
    }}
    var d0 = 0xffffffffu;
    var d1 = 0xffffffffu;
    var d2 = 0xffffffffu;
    var d3 = 0xffffffffu;
    let i0 = start + lid.x;
    if (i0 < n) {{ d0 = digit_{p}(i0); }}
    rk_digits_{p}[lid.x] = d0;
    let i1 = i0 + 256u;
    if (i1 < n) {{ d1 = digit_{p}(i1); }}
    rk_digits_{p}[lid.x + 256u] = d1;
    let i2 = i0 + 512u;
    if (i2 < n) {{ d2 = digit_{p}(i2); }}
    rk_digits_{p}[lid.x + 512u] = d2;
    let i3 = i0 + 768u;
    if (i3 < n) {{ d3 = digit_{p}(i3); }}
    rk_digits_{p}[lid.x + 768u] = d3;
    workgroupBarrier();
    if (d0 != 0xffffffffu) {{ atomicAdd(&rk_histg_{p}[(lid.x / 128u)*256u + d0], 1u); }}
    if (d1 != 0xffffffffu) {{ atomicAdd(&rk_histg_{p}[((lid.x + 256u) / 128u)*256u + d1], 1u); }}
    if (d2 != 0xffffffffu) {{ atomicAdd(&rk_histg_{p}[((lid.x + 512u) / 128u)*256u + d2], 1u); }}
    if (d3 != 0xffffffffu) {{ atomicAdd(&rk_histg_{p}[((lid.x + 768u) / 128u)*256u + d3], 1u); }}
    workgroupBarrier();
    var running = 0u;
    for (var g = 0u; g < 8u; g++) {{
        let v = atomicLoad(&rk_histg_{p}[g*256u + lid.x]);
        atomicStore(&rk_histg_{p}[g*256u + lid.x], running);
        running = running + v;
    }}
    atomicStore(&hist2d[c*256u + lid.x], running);
    workgroupBarrier();
    if (lid.x % 32u == 0u) {{
        let g = lid.x / 32u;
        for (var j = 0u; j < 128u; j++) {{
            let dd = rk_digits_{p}[g*128u + j];
            if (dd != 0xffffffffu) {{
                let lj = start + g*128u + j;
                ranks[lj] = atomicLoad(&rk_histg_{p}[g*256u + dd])
                          + rk_cntg_{p}[g*256u + dd];
                rk_cntg_{p}[g*256u + dd] = rk_cntg_{p}[g*256u + dd] + 1u;
            }}
        }}
    }}
}}

var<workgroup> cs_tile_{p}: array<u32, 8192>;

@compute @workgroup_size(256)
fn colscan_a_{p}(@builtin(workgroup_id) wgid: vec3<u32>,
                 @builtin(local_invocation_id) lid: vec3<u32>) {{
    let b = lid.x;
    let blk = wgid.x;
    let n = atomicLoad(&counts[P.c * P.tile_w * P.tile_h]);
    let n_chunks = (n + {CHUNK - 1}u) / {CHUNK}u;
    let n_blocks = (n_chunks + {T_CHUNKS - 1}u) / {T_CHUNKS}u;
    if (blk >= n_blocks) {{ return; }}
    let c0 = blk * {T_CHUNKS}u;
    for (var i = 0u; i < {T_CHUNKS}u; i++) {{
        var v = 0u;
        let c = c0 + i;
        if (c < n_chunks) {{ v = atomicLoad(&hist2d[c*256u + b]); }}
        cs_tile_{p}[i*256u + b] = v;
    }}
    workgroupBarrier();
    var acc = 0u;
    for (var i = 0u; i < {T_CHUNKS}u; i++) {{
        let idx = i*256u + b;
        let v = cs_tile_{p}[idx];
        cs_tile_{p}[idx] = acc;
        acc = acc + v;
    }}
    atomicStore(&partials[b * {maxb2}u + blk], acc);
    workgroupBarrier();
    for (var i = 0u; i < {T_CHUNKS}u; i++) {{
        let c = c0 + i;
        if (c < n_chunks) {{
            atomicStore(&hist2d[c*256u + b], cs_tile_{p}[i*256u + b]);
        }}
    }}
}}

@compute @workgroup_size(1)
fn colscan_b_{p}(@builtin(workgroup_id) wgid: vec3<u32>) {{
    let b = wgid.x;
    let n = atomicLoad(&counts[P.c * P.tile_w * P.tile_h]);
    let n_chunks = (n + {CHUNK - 1}u) / {CHUNK}u;
    let n_blocks = (n_chunks + {T_CHUNKS - 1}u) / {T_CHUNKS}u;
    var running = 0u;
    var blk = 0u;
    loop {{
        if (blk >= n_blocks) {{ break; }}
        let v = atomicLoad(&partials[b * {maxb2}u + blk]);
        atomicStore(&partials[b * {maxb2}u + blk], running);
        running = running + v;
        blk = blk + 1u;
    }}
    atomicStore(&coltot[b], running);
}}

@compute @workgroup_size(256)
fn colscan_c_{p}(@builtin(workgroup_id) wgid: vec3<u32>,
                 @builtin(local_invocation_id) lid: vec3<u32>) {{
    let b = lid.x;
    let blk = wgid.x;
    let n = atomicLoad(&counts[P.c * P.tile_w * P.tile_h]);
    let n_chunks = (n + {CHUNK - 1}u) / {CHUNK}u;
    let n_blocks = (n_chunks + {T_CHUNKS - 1}u) / {T_CHUNKS}u;
    if (blk >= n_blocks) {{ return; }}
    let base = atomicLoad(&partials[b * {maxb2}u + blk]);
    let c0 = blk * {T_CHUNKS}u;
    for (var i = 0u; i < {T_CHUNKS}u; i++) {{
        let c = c0 + i;
        if (c < n_chunks) {{
            let v = atomicLoad(&hist2d[c*256u + b]) + base;
            atomicStore(&hist2d[c*256u + b], v);
        }}
    }}
}}

var<workgroup> bs_partial_{p}: array<u32, 256>;

@compute @workgroup_size(256)
fn binscan_{p}(@builtin(local_invocation_id) lid: vec3<u32>) {{
    let T = 256u;
    let tid = u32(lid.x);
    var v = atomicLoad(&coltot[tid]);
    bs_partial_{p}[tid] = v;
    workgroupBarrier();
    var off = 1u;
    loop {{
        if (off >= T) {{ break; }}
        if (tid >= off) {{ v = v + bs_partial_{p}[tid - off]; }}
        workgroupBarrier();
        if (tid >= off) {{ bs_partial_{p}[tid] = v; }}
        workgroupBarrier();
        off = off << 1u;
    }}
    bin_base[tid] = bs_partial_{p}[tid] - atomicLoad(&coltot[tid]);
}}

@compute @workgroup_size(256)
fn scatter_{p}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let n = atomicLoad(&counts[P.c * P.tile_w * P.tile_h]);
    // grid-stride: cap can push the 1-thread-per-entry dispatch past the
    // 65535-workgroup WebGPU limit
    for (var i = gid.x; i < n; i += 16776960u) {{
        let c = i / {CHUNK}u;
        let d = digit_{p}(i);
        let pos = bin_base[d] + atomicLoad(&hist2d[c*256u + d]) + ranks[i];
        keys_{dst}[pos*2u] = keys_{src}[i*2u];
        keys_{dst}[pos*2u+1u] = keys_{src}[i*2u+1u];
        flat_{dst}[pos] = flat_{src}[i];
    }}
}}
"""


def build_wgsl(variant: str, cap: int | None = None) -> str:
    """cap: (gaussian, tile) pair capacity, baked into the isect-fill guard
    and the sort kernels' chunk-plane sizes — every Pipeline compiles a
    shader sized to its own scene. Defaults to the historical 4M."""
    cap = w2.CAP if cap is None else int(cap)
    chunks_opt = (cap + CHUNK - 1) // CHUNK
    maxb2 = (chunks_opt + T_CHUNKS - 1) // T_CHUNKS
    full = w2.WGSL.replace("const CAPACITY: u32 = 4000000u;",
                           f"const CAPACITY: u32 = {cap}u;")
    full = full.replace("const CAP_CHUNKS: u32 = 15625u;",
                        f"const CAP_CHUNKS: u32 = {(cap + 255) // 256}u;")
    head = full[:full.index("fn digit_0(")]
    tail = full[full.index("// ---- tile starts"):]
    if variant == "base":
        radix = "".join(w2.radix_kernels(p) for p in range(NPASS))
        return head + radix + tail
    elif variant == "opt":
        radix = "".join(opt_radix_kernels(p, chunks_opt) for p in range(NPASS))
    elif variant == "optcol":
        radix = "".join(optcol_radix_kernels(p, maxb2) for p in range(NPASS))
    else:
        raise ValueError(variant)
    tail_opt = tail[:tail.index("// ---- rasterize")] + FUSED_RASTERIZE
    return head + radix + tail_opt


BASE_STEPS = ("zero_hist2d", "rank", "colscan", "binscan", "scatter")
OPT_STEPS = ("count", "bin_scan", "sweep")
OPTCOL_STEPS = ("rank", "colscan_a", "colscan_b", "colscan_c", "binscan",
                "scatter")


# --------------------------------------------------------------- class ----- #

class Pipeline:
    """FullGpuPipeline clone with selectable radix variant + optional
    timestamp-query device feature + partials buffer (binding 33)."""

    def __init__(self, d, adapter_sub="", variant="opt", timestamp=False,
                 cap=None):
        """adapter_sub: substring of the Vulkan device name to pick ("nvidia",
        "5070", ...); "" takes the first Vulkan adapter. cap: (gaussian, tile)
        pair capacity -- defaults to scene-sized auto_capacity(N); pass a
        smaller value to bound scratch on memory-limited devices."""
        self.variant = variant
        self.N = int(d["xyz"].shape[0])
        self.slots = d["slots"]
        self.robot_n = int(self.slots.shape[0])
        self.scene_n = self.N - self.robot_n
        self.C = int(d["cam_pos"].shape[0])
        self.W, self.H = int(d["W"]), int(d["H"])
        self.tile_w = (self.W + TILE - 1) // TILE
        self.tile_h = (self.H + TILE - 1) // TILE
        self.nt = self.tile_w * self.tile_h
        self.CN = self.C * self.N
        self.cap = auto_capacity(self.N) if cap is None else int(cap)
        if self.cap <= 0:
            raise ValueError(f"cap must be positive, got {cap!r}")
        if variant == "base" and self.cap > w2.CAP:
            # the base radix keeps one-workgroup-per-chunk dispatches, which
            # pass the 65535-workgroup WebGPU limit beyond 4M pairs
            raise ValueError("base variant supports cap <= 4M; "
                             "use 'opt' or 'optcol' for larger scenes")
        chunks_opt = (self.cap + CHUNK - 1) // CHUNK
        base_chunks = (self.cap + 255) // 256
        maxb2 = (chunks_opt + T_CHUNKS - 1) // T_CHUNKS

        ads = [a for a in wgpu.gpu.enumerate_adapters_sync()
               if a.info["backend_type"] == "Vulkan"
               and adapter_sub.lower() in a.info["device"].lower()]
        assert ads, f"no Vulkan adapter matching {adapter_sub!r}"
        self.adapter = ads[0]
        feats = (["timestamp-query", "timestamp-query-inside-encoders",
                  "timestamp-query-inside-passes"] if timestamp else [])
        dev = self.adapter.request_device_sync(required_features=feats,
                                               required_limits={})
        self.device = dev
        q = dev.queue
        self.queue = q
        ST = wgpu.BufferUsage.STORAGE
        CS = wgpu.BufferUsage.COPY_SRC

        def put(arr):
            return dev.create_buffer_with_data(
                data=np.ascontiguousarray(arr).tobytes(), usage=ST | CS)

        def mk(nbytes):
            return dev.create_buffer(
                size=nbytes,
                usage=ST | CS | wgpu.BufferUsage.COPY_DST)

        self.b_xyz = put(d["xyz"])
        self.b_rot = put(d["rot"])
        self.b_scl = put(d["scale"])
        self.b_opa = put(d["opacity"])
        # kernels index colors as sh[g*3 + c] (degree 0 only): a full INRIA
        # PLY SH block (N,48) would silently misalign every gaussian after
        # the first -- reject anything but exactly 3 channels.
        sh = np.asarray(d["sh"])
        if sh.ndim != 2 or sh.shape[1] != 3:
            raise ValueError(
                f"d['sh'] must be (N, 3) DC coefficients, got {sh.shape}; "
                "slice a from_ply() block to sh[:, :3]")
        self.b_sh = put(sh)
        # Robot-less clouds (static scene only) carry an empty slots array;
        # wgpu rejects zero-byte buffers, so park a 4-byte placeholder.
        # update_links never reads it: with scene_n == N every gaussian takes
        # the copy path, and self_cull dispatches 0 workgroups.
        if self.robot_n:
            self.b_slots = put(self.slots)
        else:
            self.b_slots = dev.create_buffer(size=4, usage=ST | CS)
        self.b_links = dev.create_buffer(
            size=13 * 8 * 4,
            usage=ST | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC)
        self.b_vm = mk(int(np.prod(w2.make_viewmats(d).shape) * 4))
        q.write_buffer(self.b_vm, 0,
                       memoryview(np.ascontiguousarray(w2.make_viewmats(d))))
        self.b_xyz_c = mk(self.N * 3 * 4)
        self.b_rot_c = mk(self.N * 4 * 4)
        self.b_means2d = mk(self.CN * 2 * 4)
        self.b_depths = mk(self.CN * 4)
        self.b_conics = mk(self.CN * 3 * 4)
        self.b_radii = mk(self.CN * 2 * 4)
        self.b_colors = mk(self.CN * 4 * 4)
        self.b_opacity_b = mk(self.CN * 4)
        self.b_ranges = mk(self.CN * 4 * 4)
        self.b_counts = mk((self.C * self.nt + 1) * 4)
        self.b_offsets = mk((self.C * self.nt + 1) * 4)
        self.b_cursors = mk((self.C * self.nt + 1) * 4)
        self.b_keys_a = mk(self.cap * 2 * 4)
        self.b_flat_a = mk(self.cap * 4)
        self.b_keys_b = mk(self.cap * 2 * 4)
        self.b_flat_b = mk(self.cap * 4)
        self.b_hist2d = mk(base_chunks * 256 * 4)  # base variant zero range
        self.b_starts = mk((self.C * self.nt + 1) * 4)
        self.b_rgb = mk(self.C * self.H * self.W * 4)
        self.b_depth = mk(self.C * self.H * self.W * 4)
        self.b_renders = mk(self.C * self.H * self.W * 4 * 4)
        self.b_ranks = mk(self.cap * 4)
        self.b_coltot = mk(256 * 4)
        self.b_bin_base = mk(256 * 4)
        self.b_partials = mk(maxb2 * 256 * 4)
        self.b_status = mk(NPASS * chunks_opt * 256 * 4)
        self.b_bin_cnt = mk(NPASS * 256 * 4)

        u = np.zeros(16, np.uint32)
        u[:8] = (self.N, self.C, self.W, self.H, self.tile_w,
                 self.tile_h, self.scene_n, self.robot_n)
        fxy, cx, cy = w2.make_K(d)
        self._K = (fxy, cx, cy)
        cpos0 = d["cam_pos"][0]
        u[8:16] = np.array([fxy, cx, cy, np.float32(0.30),
                            cpos0[0], cpos0[1], cpos0[2], 0.0],
                           np.float32).view(np.uint32)
        self.b_params = dev.create_buffer(
            size=64, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        q.write_buffer(self.b_params, 0, memoryview(u))

        self.bufs = [self.b_params, self.b_xyz, self.b_rot, self.b_scl,
                     self.b_opa, self.b_sh, self.b_slots, self.b_links,
                     self.b_vm, self.b_xyz_c, self.b_rot_c, self.b_means2d,
                     self.b_depths, self.b_conics, self.b_radii,
                     self.b_colors, self.b_opacity_b, self.b_ranges,
                     self.b_counts, self.b_offsets, self.b_cursors,
                     self.b_keys_a, self.b_flat_a, self.b_keys_b,
                     self.b_flat_b, self.b_hist2d, self.b_starts, self.b_rgb,
                     self.b_depth, self.b_renders, self.b_ranks,
                     self.b_coltot, self.b_bin_base, self.b_partials,
                     self.b_status, self.b_bin_cnt]
        atomic = {18, 19, 20, 25, 27, 28, 31, 33, 34, 35}
        readonly = {1, 2, 3, 5, 6, 7, 8}
        self.bgl = dev.create_bind_group_layout(entries=[
            {"binding": i, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": "uniform" if i == 0 else
                        ("read-only-storage" if i in readonly else "storage")}}
            for i in range(len(self.bufs))])
        self.bg = dev.create_bind_group(layout=self.bgl, entries=[
            {"binding": i, "resource": {"buffer": b, "offset": 0,
                                        "size": b.size}}
            for i, b in enumerate(self.bufs)])
        module = dev.create_shader_module(code=build_wgsl(variant, cap=self.cap))
        pl = dev.create_pipeline_layout(bind_group_layouts=[self.bgl])

        def pipe(entry):
            return dev.create_compute_pipeline(
                layout=pl, compute={"module": module, "entry_point": entry})

        names = ["update_links", "self_cull", "project", "isect_count",
                 "zero_counts", "scan_counts", "isect_fill",
                 "tile_starts", "rasterize"]
        if variant == "base":
            names.append("quantize")
        elif variant == "opt":
            names.append("zero_sweep")
        steps = (BASE_STEPS if variant == "base" else
                 OPT_STEPS if variant == "opt" else OPTCOL_STEPS)
        for p in range(NPASS):
            names.extend(f"{s}_{p}" for s in steps)
        self.pipes = {name: pipe(name) for name in names}

        self.wg = {
            "update_links": ((self.N + 255) // 256, 1, 1),
            "self_cull": ((self.robot_n + 255) // 256, 1, 1),
            "project": ((self.CN + 255) // 256, 1, 1),
            "isect_count": ((self.CN + 255) // 256, 1, 1),
            "zero_counts": ((self.C * self.nt + 1 + 255) // 256, 1, 1),
            "scan_counts": (1, 1, 1),
            "isect_fill": ((self.CN + 255) // 256, 1, 1),
            "tile_starts": ((self.C * self.nt + 1 + 255) // 256, 1, 1),
            "rasterize": (self.C * self.nt, 1, 1),
        }
        if variant == "base":
            self.wg["quantize"] = ((self.C * self.H * self.W + 255) // 256,
                                   1, 1)
        for p in range(NPASS):
            if variant == "optcol":
                self.wg[f"rank_{p}"] = (chunks_opt, 1, 1)
                self.wg[f"colscan_a_{p}"] = (maxb2, 1, 1)
                self.wg[f"colscan_b_{p}"] = (256, 1, 1)
                self.wg[f"colscan_c_{p}"] = (maxb2, 1, 1)
                self.wg[f"binscan_{p}"] = (1, 1, 1)
                self.wg[f"scatter_{p}"] = (min((self.cap + 255) // 256,
                                               65535), 1, 1)
            elif variant == "base":
                self.wg[f"zero_hist2d_{p}"] = (min(base_chunks, 65535), 1, 1)
                self.wg[f"rank_{p}"] = (base_chunks, 1, 1)
                self.wg[f"colscan_{p}"] = (256, 1, 1)
                self.wg[f"binscan_{p}"] = (1, 1, 1)
                self.wg[f"scatter_{p}"] = (min((self.cap + 255) // 256,
                                               65535), 1, 1)
            else:
                self.wg[f"count_{p}"] = (chunks_opt, 1, 1)
                self.wg[f"bin_scan_{p}"] = (1, 1, 1)
                self.wg[f"sweep_{p}"] = (chunks_opt, 1, 1)
        if variant == "opt":
            self.wg["zero_sweep"] = (
                min((NPASS * chunks_opt * 256 + NPASS * 256 + 255) // 256,
                    65535), 1, 1)
        tail = ["tile_starts", "rasterize"]
        if variant == "base":
            tail.append("quantize")
        head = ["update_links", "self_cull", "project", "zero_counts",
                "isect_count", "scan_counts", "isect_fill"]
        if variant == "opt":
            head.append("zero_sweep")
        self.order = head + [f"{s}_{p}" for p in range(NPASS) for s in steps] + tail

    def set_links(self, pos, quat):
        links = np.zeros((13, 8), np.float32)
        links[:, :3] = pos
        links[:, 3:7] = quat
        self.queue.write_buffer(self.b_links, 0, memoryview(links.reshape(-1)))

    def set_cam(self, cam_pos, cam_xmat):
        """Update camera poses (C,3)/(C,3,3) each frame: rebuild viewmats
        and the cull origin so the render follows a moving base."""
        d = {"cam_pos": np.asarray(cam_pos, np.float32).reshape(-1, 3),
             "cam_xmat": np.asarray(cam_xmat, np.float32).reshape(-1, 9)}
        self.queue.write_buffer(
            self.b_vm, 0,
            memoryview(np.ascontiguousarray(w2.make_viewmats(d))))
        u = np.zeros(16, np.uint32)
        u[:8] = (self.N, self.C, self.W, self.H, self.tile_w,
                 self.tile_h, self.scene_n, self.robot_n)
        fxy, cx, cy = self._K
        p0 = d["cam_pos"][0]
        u[8:16] = np.array([fxy, cx, cy, np.float32(0.30),
                            p0[0], p0[1], p0[2], 0.0],
                           np.float32).view(np.uint32)
        self.queue.write_buffer(self.b_params, 0, memoryview(u))

    def render_frame(self):
        enc = self.device.create_command_encoder()
        cp = enc.begin_compute_pass()
        for name in self.order:
            cp.set_pipeline(self.pipes[name])
            cp.set_bind_group(0, self.bg)
            cp.dispatch_workgroups(*self.wg[name])
        cp.end()
        self.queue.submit([enc.finish()])
        # capacity tripwire: scan_counts parks the TRUE pair total in counts'
        # last cell even when isect_fill had to drop pairs past cap -- raise
        # instead of shipping a frame with silent speckle holes.
        n_pairs = int(np.frombuffer(self.queue.read_buffer(
            self.b_counts, self.b_counts.size - 4, 4).cast("I"),
            np.uint32)[0])
        if n_pairs > self.cap:
            raise RuntimeError(
                f"isect overflow: {n_pairs:,} (gaussian, tile) pairs exceed "
                f"cap {self.cap:,} -- raise Pipeline(cap=...) or subsample "
                "the cloud")
        rgb = np.frombuffer(self.queue.read_buffer(self.b_rgb).cast("I"),
                            np.uint32)
        depth = np.frombuffer(self.queue.read_buffer(self.b_depth).cast("I"),
                              np.uint32)
        return (rgb.reshape(self.C, self.H, self.W),
                depth.reshape(self.C, self.H, self.W))

    def unpack(self, rgb_packed, depth_mm):
        r = (rgb_packed & np.uint32(0xFF)).astype(np.uint8)
        g = ((rgb_packed >> np.uint32(8)) & np.uint32(0xFF)).astype(np.uint8)
        b = ((rgb_packed >> np.uint32(16)) & np.uint32(0xFF)).astype(np.uint8)
        return np.stack([r, g, b], axis=-1), depth_mm.astype(np.uint16)


# --------------------------------------------------------------- main ------ #

def pick_variant(sel: str) -> str:
    """OneSweep look-back deadlocks Mesa/Intel (device lost even at 2
    blocks); NVIDIA verified. Everything non-NVIDIA gets the hierarchical
    colscan variant (all other optimizations kept)."""
    ads = [a for a in wgpu.gpu.enumerate_adapters_sync()
           if a.info["backend_type"] == "Vulkan"
           and sel.lower() in a.info["device"].lower()]
    vend = ads[0].info["vendor"].lower() if ads else ""
    return "optcol" if "nvidia" not in vend else "opt"



__all__ = ["Pipeline", "pick_variant", "build_wgsl", "make_K", "auto_capacity",
           "CAP", "CAP_MAX", "TILE", "NPASS"]
