"""WGSL base + FullGpuPipeline promoted from examples/wgpu/w2_full_pipe.py
(P2 acceptance artifact, frozen). gausscam.backends.webgpu splices its WGSL
and constants; the example file remains as the historical benchmark CLI.
"""
import numpy as np
import wgpu

TILE = 16
CAP = 4_000_000          # isect capacity (entries)
N_CHUNKS = (CAP + 255) // 256
NPASS = 6                # 8-bit digits: 4x lo + 2x hi

WGSL = """
const SH0: f32 = 0.2820947917738781;
const ALPHA_THRESHOLD: f32 = 0.00392156886;  // 1/255

struct Params {
    n: u32, c: u32, w: u32, h: u32,
    tile_w: u32, tile_h: u32, scene_n: u32, robot_n: u32,
    fxy: f32, cx: f32, cy: f32, cull_r: f32,
    cox: f32, coy: f32, coz: f32, pad: f32,
};
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> xyz_o: array<f32>;
@group(0) @binding(2) var<storage, read> rot_o: array<f32>;
@group(0) @binding(3) var<storage, read> scl: array<f32>;
@group(0) @binding(4) var<storage, read_write> opa: array<f32>;
@group(0) @binding(5) var<storage, read> sh: array<f32>;
@group(0) @binding(6) var<storage, read> slots: array<i32>;
@group(0) @binding(7) var<storage, read> links: array<f32>;      // [n_links*8] pos3+quat_wxyz4+pad
@group(0) @binding(8) var<storage, read> viewmats: array<f32>;
@group(0) @binding(9) var<storage, read_write> xyz_c: array<f32>;
@group(0) @binding(10) var<storage, read_write> rot_c: array<f32>;
@group(0) @binding(11) var<storage, read_write> means2d: array<f32>;
@group(0) @binding(12) var<storage, read_write> depths: array<f32>;
@group(0) @binding(13) var<storage, read_write> conics: array<f32>;
@group(0) @binding(14) var<storage, read_write> radii: array<i32>;
@group(0) @binding(15) var<storage, read_write> colors: array<f32>;
@group(0) @binding(16) var<storage, read_write> opacity_b: array<f32>;
@group(0) @binding(17) var<storage, read_write> ranges: array<u32>;   // [4*CN] x0,y0,w,h
@group(0) @binding(18) var<storage, read_write> counts: array<atomic<u32>>;   // [C*nt+1]
@group(0) @binding(19) var<storage, read_write> offsets: array<atomic<u32>>;  // [C*nt+1]
@group(0) @binding(20) var<storage, read_write> cursors: array<atomic<u32>>;  // [C*nt+1]
@group(0) @binding(21) var<storage, read_write> keys_a: array<u32>;   // [2*CAP] hi,lo interleave
@group(0) @binding(22) var<storage, read_write> flat_a: array<i32>;
@group(0) @binding(23) var<storage, read_write> keys_b: array<u32>;
@group(0) @binding(24) var<storage, read_write> flat_b: array<i32>;
@group(0) @binding(25) var<storage, read_write> hist2d: array<atomic<u32>>;   // [N_CHUNKS*256]
@group(0) @binding(26) var<storage, read_write> starts: array<i32>;           // [C*nt+1]
@group(0) @binding(27) var<storage, read_write> rgb_out: array<atomic<u32>>;
@group(0) @binding(28) var<storage, read_write> depth_out: array<atomic<u32>>;
@group(0) @binding(29) var<storage, read_write> renders: array<f32>;
@group(0) @binding(30) var<storage, read_write> ranks: array<u32>;            // [CAP]
@group(0) @binding(31) var<storage, read_write> coltot: array<atomic<u32>>;   // [256]
@group(0) @binding(32) var<storage, read_write> bin_base: array<u32>;         // [256]

fn tile_bits() -> u32 {
    let nt = P.tile_w * P.tile_h;
    return u32(firstLeadingBit(i32(nt)) + 1);  // floor(log2(nt))+1, CUDA formula
}

// ---- pose refresh: copy scene block, rigid-transform robot block ----
@compute @workgroup_size(256)
fn update_links(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    if (g >= P.n) { return; }
    if (g < P.scene_n) {
        xyz_c[g*3u] = xyz_o[g*3u];
        xyz_c[g*3u+1u] = xyz_o[g*3u+1u];
        xyz_c[g*3u+2u] = xyz_o[g*3u+2u];
        rot_c[g*4u] = rot_o[g*4u];
        rot_c[g*4u+1u] = rot_o[g*4u+1u];
        rot_c[g*4u+2u] = rot_o[g*4u+2u];
        rot_c[g*4u+3u] = rot_o[g*4u+3u];
        return;
    }
    let k = g - P.scene_n;
    let b = u32(slots[k]);
    let lp = vec3<f32>(links[b*8u], links[b*8u+1u], links[b*8u+2u]);
    // link quat stored wxyz in the buffer: +3=w, +4=x, +5=y, +6=z
    let w1 = links[b*8u+3u];
    let x1 = links[b*8u+4u];
    let y1 = links[b*8u+5u];
    let z1 = links[b*8u+6u];
    let p = vec3<f32>(xyz_o[g*3u], xyz_o[g*3u+1u], xyz_o[g*3u+2u]);
    let u = vec3<f32>(x1, y1, z1);
    let uv = cross(u, p);
    let uuv = cross(u, uv);
    let pr = p + 2.0 * (w1 * uv + uuv);
    xyz_c[g*3u] = pr.x + lp.x;
    xyz_c[g*3u+1u] = pr.y + lp.y;
    xyz_c[g*3u+2u] = pr.z + lp.z;
    let r = vec4<f32>(rot_o[g*4u], rot_o[g*4u+1u], rot_o[g*4u+2u], rot_o[g*4u+3u]);
    let w2 = r.w; let x2 = r.x; let y2 = r.y; let z2 = r.z;
    rot_c[g*4u] = w1*w2 - x1*x2 - y1*y2 - z1*z2;
    rot_c[g*4u+1u] = w1*x2 + x1*w2 + y1*z2 - z1*y2;
    rot_c[g*4u+2u] = w1*y2 - x1*z2 + y1*w2 + z1*x2;
    rot_c[g*4u+3u] = w1*z2 + x1*y2 - y1*x2 + z1*w2;
}

// ---- self cull: zero opacity of robot gaussians near the origin ----
@compute @workgroup_size(256)
fn self_cull(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if (k >= P.robot_n) { return; }
    let g = P.scene_n + k;
    let x = xyz_c[g*3u] - P.cox;
    let y = xyz_c[g*3u+1u] - P.coy;
    let z = xyz_c[g*3u+2u] - P.coz;
    let d = sqrt(x*x + z*z + y*y);
    if (d < P.cull_r) {
        opa[g] = 0.0;
    }
}

// ---- project + degree-0 colors + opacity broadcast (validated in W1) ----
@compute @workgroup_size(256)
fn project(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= P.c * P.n) { return; }
    let g = idx % P.n;
    let cam = idx / P.n;
    let vm = cam * 16u;

    var out_x = 0.0; var out_y = 0.0;
    var out_c0 = 0.0; var out_c1 = 0.0; var out_c2 = 0.0;
    var out_depth = 0.0;
    var rx = 0i; var ry = 0i;

    let p = vec3<f32>(xyz_c[g*3u], xyz_c[g*3u+1u], xyz_c[g*3u+2u]);
    let r0 = vec3<f32>(viewmats[vm], viewmats[vm+1u], viewmats[vm+2u]);
    let r1 = vec3<f32>(viewmats[vm+4u], viewmats[vm+5u], viewmats[vm+6u]);
    let r2 = vec3<f32>(viewmats[vm+8u], viewmats[vm+9u], viewmats[vm+10u]);
    let t = vec3<f32>(viewmats[vm+3u], viewmats[vm+7u], viewmats[vm+11u]);
    let mean_c = vec3<f32>(dot(r0, p) + t.x, dot(r1, p) + t.y, dot(r2, p) + t.z);

    if (mean_c.z >= 0.01 && mean_c.z <= 1e10) {
        let qw = rot_c[g*4u]; let qx = rot_c[g*4u+1u]; let qy = rot_c[g*4u+2u]; let qz = rot_c[g*4u+3u];
        let inorm = inverseSqrt(qx*qx + qy*qy + qz*qz + qw*qw);
        let x = qx * inorm; let y = qy * inorm; let z = qz * inorm; let w = qw * inorm;
        let x2 = x*x; let y2 = y*y; let z2 = z*z;
        let xy = x*y; let xz = x*z; let yz = y*z;
        let wx = w*x; let wy = w*y; let wz = w*z;
        let m00 = 1.0 - 2.0*(y2+z2); let m10 = 2.0*(xy+wz); let m20 = 2.0*(xz-wy);
        let m01 = 2.0*(xy-wz); let m11 = 1.0 - 2.0*(x2+z2); let m21 = 2.0*(yz+wx);
        let m02 = 2.0*(xz+wy); let m12 = 2.0*(yz-wx); let m22 = 1.0 - 2.0*(x2+y2);
        let s0 = scl[g*3u]; let s1 = scl[g*3u+1u]; let s2 = scl[g*3u+2u];
        let q0 = s0*s0; let q1 = s1*s1; let q2 = s2*s2;
        let cv00 = m00*m00*q0 + m01*m01*q1 + m02*m02*q2;
        let cv01 = m00*m10*q0 + m01*m11*q1 + m02*m12*q2;
        let cv02 = m00*m20*q0 + m01*m21*q1 + m02*m22*q2;
        let cv11 = m10*m10*q0 + m11*m11*q1 + m12*m12*q2;
        let cv12 = m10*m20*q0 + m11*m21*q1 + m12*m22*q2;
        let cv22 = m20*m20*q0 + m21*m21*q1 + m22*m22*q2;
        let cv0 = vec3<f32>(cv00, cv01, cv02);
        let cv1 = vec3<f32>(cv01, cv11, cv12);
        let cv2 = vec3<f32>(cv02, cv12, cv22);
        // covar_c = R cv R^T: a_r = row r of (R cv); cv symmetric
        let a0 = vec3<f32>(dot(r0, cv0), dot(r0, cv1), dot(r0, cv2));
        let a1 = vec3<f32>(dot(r1, cv0), dot(r1, cv1), dot(r1, cv2));
        let a2 = vec3<f32>(dot(r2, cv0), dot(r2, cv1), dot(r2, cv2));
        let cc00 = dot(a0, r0); let cc01 = dot(a0, r1); let cc02 = dot(a0, r2);
        let cc10 = dot(a1, r0); let cc11 = dot(a1, r1); let cc12 = dot(a1, r2);
        let cc20 = dot(a2, r0); let cc21 = dot(a2, r1); let cc22 = dot(a2, r2);
        let cc0 = vec3<f32>(cc00, cc01, cc02);
        let cc1 = vec3<f32>(cc10, cc11, cc12);
        let cc2 = vec3<f32>(cc20, cc21, cc22);

        let fx = P.fxy; let fy = P.fxy; let cxk = P.cx; let cyk = P.cy;
        let x3 = mean_c.x; let y3 = mean_c.y; let z3 = mean_c.z;
        let tan_fovx = 0.5 * f32(P.w) / fx;
        let tan_fovy = 0.5 * f32(P.h) / fy;
        let lim_x_pos = (f32(P.w) - cxk) / fx + 0.3 * tan_fovx;
        let lim_x_neg = cxk / fx + 0.3 * tan_fovx;
        let lim_y_pos = (f32(P.h) - cyk) / fy + 0.3 * tan_fovy;
        let lim_y_neg = cyk / fy + 0.3 * tan_fovy;
        let rz = 1.0 / z3;
        let rz2 = rz * rz;
        let tx = z3 * min(lim_x_pos, max(-lim_x_neg, x3 * rz));
        let ty = z3 * min(lim_y_pos, max(-lim_y_neg, y3 * rz));
        let J0 = vec3<f32>(fx * rz, 0.0, -fx * tx * rz2);
        let J1 = vec3<f32>(0.0, fy * rz, -fy * ty * rz2);
        let ccj0 = cc0 * J0.x + cc1 * J0.y + cc2 * J0.z;
        let ccj1 = cc0 * J1.x + cc1 * J1.y + cc2 * J1.z;
        let v00 = dot(J0, ccj0);
        let v01 = dot(J0, ccj1);
        let v11 = dot(J1, ccj1);
        let mean2d = vec2<f32>(fx * x3 * rz + cxk, fy * y3 * rz + cyk);

        let a00 = v00 + 0.3;
        let a11 = v11 + 0.3;
        let det = a00 * a11 - v01 * v01;
        if (det > 0.0) {
            let op = opa[g];
            if (op >= ALPHA_THRESHOLD) {
                let extend = min(3.33, sqrt(2.0 * log(op / ALPHA_THRESHOLD)));
                let rxi = i32(ceil(extend * sqrt(a00)));
                let ryi = i32(ceil(extend * sqrt(a11)));
                let ok = !(mean2d.x + f32(rxi) <= 0.0 || mean2d.x - f32(rxi) >= f32(P.w)
                        || mean2d.y + f32(ryi) <= 0.0 || mean2d.y - f32(ryi) >= f32(P.h));
                if (rxi > 0 && ryi > 0 && ok) {
                    out_x = mean2d.x; out_y = mean2d.y;
                    out_c0 = a11 / det;
                    out_c1 = -v01 / det;
                    out_c2 = a00 / det;
                    out_depth = mean_c.z;
                    rx = rxi; ry = ryi;
                }
            }
        }
    }

    means2d[idx*2u] = out_x;
    means2d[idx*2u+1u] = out_y;
    conics[idx*3u] = out_c0;
    conics[idx*3u+1u] = out_c1;
    conics[idx*3u+2u] = out_c2;
    depths[idx] = out_depth;
    radii[idx*2u] = rx;
    radii[idx*2u+1u] = ry;
    var col = vec3<f32>(0.5, 0.5, 0.5);
    if (rx > 0) {
        col = max(vec3<f32>(SH0 * sh[g*3u] + 0.5, SH0 * sh[g*3u+1u] + 0.5,
                            SH0 * sh[g*3u+2u] + 0.5), vec3<f32>(0.0, 0.0, 0.0));
    }
    colors[idx*4u] = col.x;
    colors[idx*4u+1u] = col.y;
    colors[idx*4u+2u] = col.z;
    colors[idx*4u+3u] = out_depth;
    opacity_b[idx] = opa[g];
}

// ---- isect count: tile ranges + per-tile atomic counts ----
@compute @workgroup_size(256)
fn isect_count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= P.c * P.n) { return; }
    let rx = radii[idx*2u];
    let ry = radii[idx*2u+1u];
    if (rx <= 0 || ry <= 0) {
        ranges[idx*4u] = 0u; ranges[idx*4u+1u] = 0u;
        ranges[idx*4u+2u] = 0u; ranges[idx*4u+3u] = 0u;
        return;
    }
    let trx = f32(rx) / 16.0;
    let trY = f32(ry) / 16.0;
    let tx = means2d[idx*2u] / 16.0;
    let ty = means2d[idx*2u+1u] / 16.0;
    let x0 = u32(clamp(floor(tx - trx), 0.0, f32(P.tile_w)));
    let y0 = u32(clamp(floor(ty - trY), 0.0, f32(P.tile_h)));
    let x1 = u32(clamp(ceil(tx + trx), 0.0, f32(P.tile_w)));
    let y1 = u32(clamp(ceil(ty + trY), 0.0, f32(P.tile_h)));
    let w_ = x1 - x0;
    let h_ = y1 - y0;
    ranges[idx*4u] = x0;
    ranges[idx*4u+1u] = y0;
    ranges[idx*4u+2u] = w_;
    ranges[idx*4u+3u] = h_;
    let cam = idx / P.n;
    let base = cam * (P.tile_w * P.tile_h);
    var iy = 0u;
    loop {
        if (iy >= h_) { break; }
        var ix = 0u;
        loop {
            if (ix >= w_) { break; }
            atomicAdd(&counts[base + (y0 + iy) * P.tile_w + (x0 + ix)], 1u);
            ix = ix + 1u;
        }
        iy = iy + 1u;
    }
}

// ---- zero counts (must run before isect_count each frame) ----
@compute @workgroup_size(256)
fn zero_counts(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i > P.c * P.tile_w * P.tile_h) { return; }
    atomicStore(&counts[i], 0u);
}

var<workgroup> scan_partial: array<u32, 256>;

// ---- exclusive scan of counts -> offsets + cursors + total ----
@compute @workgroup_size(256)
fn scan_counts(@builtin(local_invocation_id) lid: vec3<u32>) {
    let T = 256u;
    let tid = u32(lid.x);
    let n_plus = P.c * P.tile_w * P.tile_h + 1u;
    var running = 0u;
    var base = 0u;
    loop {
        if (base >= n_plus) { break; }
        let i = base + tid;
        var v = 0u;
        if (i < n_plus) { v = atomicLoad(&counts[i]); }
        scan_partial[tid] = v;
        workgroupBarrier();
        var off = 1u;
        loop {
            if (off >= T) { break; }
            if (tid >= off) { v = v + scan_partial[tid - off]; }
            workgroupBarrier();
            if (tid >= off) { scan_partial[tid] = v; }
            workgroupBarrier();
            off = off << 1u;
        }
        if (i < n_plus) {
            let excl = running + scan_partial[tid] - atomicLoad(&counts[i]);
            atomicStore(&offsets[i], excl);
            atomicStore(&cursors[i], excl);
        }
        running = running + scan_partial[T - 1u];
        workgroupBarrier();
        base = base + T;
    }
    if (tid == 0u) {
        atomicStore(&counts[n_plus - 1u], running);  // total isects
    }
}

// ---- fill: expand ranges into (key, flat) pairs ----
@compute @workgroup_size(256)
fn isect_fill(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= P.c * P.n) { return; }
    let w_ = ranges[idx*4u+2u];
    let h_ = ranges[idx*4u+3u];
    if (w_ == 0u || h_ == 0u) { return; }
    let x0 = ranges[idx*4u];
    let y0 = ranges[idx*4u+1u];
    let cam = idx / P.n;
    let bits = tile_bits();
    let lo = bitcast<u32>(depths[idx]);
    let base = cam * (P.tile_w * P.tile_h);
    var iy = 0u;
    loop {
        if (iy >= h_) { break; }
        var ix = 0u;
        loop {
            if (ix >= w_) { break; }
            let tid = base + (y0 + iy) * P.tile_w + (x0 + ix);
            let pos = atomicAdd(&cursors[tid], 1u);
            if (pos < 4000000u) {
                keys_a[pos*2u] = (cam << bits) | (tid - base);
                keys_a[pos*2u+1u] = lo;
                flat_a[pos] = i32(idx);
            }
            ix = ix + 1u;
        }
        iy = iy + 1u;
    }
}
"""


def radix_kernels(p: int) -> str:
    """STABLE 8-bit-digit pass: zero hist2d -> rank (in-chunk,
    order-preserving) -> per-chunk column scan -> bin base scan -> scatter."""
    # keys layout: keys[2i] = hi, keys[2i+1] = lo. Digits LSB-first:
    # p0..p3 = lo bytes, p4..p5 = hi bytes.
    word_sel = "1u" if p < 4 else "0u"
    shift = 8 * (p % 4)
    src = "a" if p % 2 == 0 else "b"
    dst = "b" if p % 2 == 0 else "a"
    return f"""
fn digit_{p}(i: u32) -> u32 {{
    let word = keys_{src}[i*2u + {word_sel}];
    return (word >> {shift}u) & 0xffu;
}}

// hist2d must be zeroed each pass (rank atomicAdds into it)
@compute @workgroup_size(256)
fn zero_hist2d_{p}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if (gid.x < 15625u * 256u) {{ atomicStore(&hist2d[gid.x], 0u); }}
}}

// per-256-chunk: shared digits, in-order local ranks, chunk histogram
var<workgroup> rk_digits_{p}: array<u32, 256>;

@compute @workgroup_size(256)
fn rank_{p}(@builtin(workgroup_id) wgid: vec3<u32>,
            @builtin(local_invocation_id) lid: vec3<u32>) {{
    let c = wgid.x;
    let n = atomicLoad(&counts[P.c * P.tile_w * P.tile_h]);
    let start = c * 256u;
    if (start >= n) {{ return; }}
    let i = start + lid.x;
    let in_range = i < n;
    var d = 0u;
    if (in_range) {{
        d = digit_{p}(i);
        rk_digits_{p}[lid.x] = d;
    }}
    workgroupBarrier();
    if (in_range) {{
        var local_rank = 0u;
        var j = 0u;
        loop {{
            if (j >= lid.x) {{ break; }}
            if (rk_digits_{p}[j] == d) {{ local_rank = local_rank + 1u; }}
            j = j + 1u;
        }}
        ranks[i] = local_rank;
        atomicAdd(&hist2d[c*256u + d], 1u);
    }}
}}

// per-bin exclusive scan across chunks: hist2d[c][b] = sum_(c'<c)
@compute @workgroup_size(1)
fn colscan_{p}(@builtin(workgroup_id) wgid: vec3<u32>) {{
    let b = wgid.x;
    let n = atomicLoad(&counts[P.c * P.tile_w * P.tile_h]);
    let n_chunks = (n + 255u) / 256u;
    var running = 0u;
    var c = 0u;
    loop {{
        if (c >= n_chunks) {{ break; }}
        let v = atomicExchange(&hist2d[c*256u + b], running);
        running = running + v;
        c = c + 1u;
    }}
    atomicStore(&coltot[b], running);
}}

// exclusive scan of the 256 bin totals -> bin bases
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
    let i = gid.x;
    let n = atomicLoad(&counts[P.c * P.tile_w * P.tile_h]);
    if (i >= n) {{ return; }}
    let c = i / 256u;
    let d = digit_{p}(i);
    let pos = bin_base[d] + hist2d[c*256u + d] + ranks[i];
    keys_{dst}[pos*2u] = keys_{src}[i*2u];
    keys_{dst}[pos*2u+1u] = keys_{src}[i*2u+1u];
    flat_{dst}[pos] = flat_{src}[i];
}}
"""

WGSL += "".join(radix_kernels(p) for p in range(NPASS))

WGSL += """
// ---- tile starts: lower_bound of each (cam,tile) hi in sorted keys ----
@compute @workgroup_size(256)
fn tile_starts(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n_tiles = P.tile_w * P.tile_h;
    let n_plus = P.c * n_tiles + 1u;
    if (i >= n_plus) { return; }
    if (i == n_plus - 1u) {
        starts[i] = i32(atomicLoad(&counts[n_plus - 1u]));
        return;
    }
    let bits = tile_bits();
    let tgt = ((i / n_tiles) << bits) | (i % n_tiles);
    var lo = 0u;
    var hi = atomicLoad(&counts[n_plus - 1u]);
    loop {
        if (lo >= hi) { break; }
        let mid = (lo + hi) / 2u;
        if (keys_a[mid*2u] < tgt) { lo = mid + 1u; }
        else { hi = mid; }
    }
    starts[i] = i32(lo);
}

// ---- rasterize (validated in W1) ----
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
        renders[pix_id*4u] = pix_out.x;
        renders[pix_id*4u+1u] = pix_out.y;
        renders[pix_id*4u+2u] = pix_out.z;
        renders[pix_id*4u+3u] = pix_out.w;
    }
}

// ---- quantize: f32 renders -> packed u8 rgb + u16 mm depth (u32 each) ----
@compute @workgroup_size(256)
fn quantize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pix = gid.x;
    if (pix >= P.c * P.h * P.w) { return; }
    let r = renders[pix*4u];
    let g = renders[pix*4u+1u];
    let b = renders[pix*4u+2u];
    let ru = u32(clamp(r, 0.0, 1.0) * 255.0);
    let gu = u32(clamp(g, 0.0, 1.0) * 255.0);
    let bu = u32(clamp(b, 0.0, 1.0) * 255.0);
    atomicStore(&rgb_out[pix], (bu << 16u) | (gu << 8u) | ru);
    let dv = clamp(renders[pix*4u+3u], 0.0, 65.535) * 1000.0;
    let vv = floor(dv);
    let fr = dv - vv;
    // round-half-even at exact .5 (matches CUDA rintf)
    let res = select(vv + 1.0, vv, (fr < 0.5) || (fr == 0.5 && (u32(vv) & 1u) == 0u));
    atomicStore(&depth_out[pix], u32(res));
}
"""


def make_viewmats(d):
    cpos, cxm = d["cam_pos"], d["cam_xmat"]
    C = cpos.shape[0]
    T = np.zeros((C, 4, 4), np.float32)
    T[:, :3, :3] = cxm.reshape(C, 3, 3)
    T[:, :3, 3] = cpos
    T[:, 3, 3] = 1.0
    T[:, :, 1:3] *= -1.0
    vm = np.linalg.inv(T).astype(np.float32)
    return vm.reshape(C, -1)


def make_K(d):
    H, W = int(d["H"]), int(d["W"])
    fovy = np.float32(d["fovy"])
    rad = np.float32(np.float64(fovy) * np.float64(0.017453292519943295))
    tan_half = np.float32(np.tan(np.float32(rad * np.float32(0.5))))
    fxy = np.float32(np.float32(H) / (np.float32(2.0) * tan_half))
    return fxy, np.float32(W / 2.0), np.float32(H / 2.0)


class FullGpuPipeline:
    """Upload once, then render frames with a single encoder/submit."""

    def __init__(self, d, adapter_sub="5070"):
        self.d = d
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

        ads = [a for a in wgpu.gpu.enumerate_adapters_sync()
               if a.info["backend_type"] == "Vulkan"
               and adapter_sub.lower() in a.info["device"].lower()]
        assert ads, f"no Vulkan adapter matching {adapter_sub!r}"
        self.adapter = ads[0]
        dev = self.adapter.request_device_sync(
            required_features=[], required_limits={})
        self.device = dev
        q = dev.queue
        self.queue = q
        ST = wgpu.BufferUsage.STORAGE
        CS = wgpu.BufferUsage.COPY_SRC

        def put(arr):
            return dev.create_buffer_with_data(
                data=np.ascontiguousarray(arr).tobytes(), usage=ST | CS)

        def mk(nbytes):
            return dev.create_buffer(size=nbytes, usage=ST | CS)

        self.b_xyz = put(d["xyz"])
        self.b_rot = put(d["rot"])
        self.b_scl = put(d["scale"])
        self.b_opa = put(d["opacity"])  # cull zeroes in place
        self.b_sh = put(d["sh"])
        self.b_slots = put(self.slots)
        self.b_links = dev.create_buffer(
            size=13 * 8 * 4,
            usage=ST | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC)
        self.b_vm = put(make_viewmats(d))
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
        self.b_keys_a = mk(CAP * 2 * 4)
        self.b_flat_a = mk(CAP * 4)
        self.b_keys_b = mk(CAP * 2 * 4)
        self.b_flat_b = mk(CAP * 4)
        self.b_hist2d = mk(N_CHUNKS * 256 * 4)
        self.b_starts = mk((self.C * self.nt + 1) * 4)
        self.b_rgb = mk(self.C * self.H * self.W * 4)
        self.b_depth = mk(self.C * self.H * self.W * 4)
        self.b_renders = mk(self.C * self.H * self.W * 4 * 4)
        self.b_ranks = mk(CAP * 4)
        self.b_coltot = mk(256 * 4)
        self.b_bin_base = mk(256 * 4)

        self.u = np.zeros(16, np.uint32)
        self.u[:8] = (self.N, self.C, self.W, self.H, self.tile_w,
                      self.tile_h, self.scene_n, self.robot_n)
        fxy, cx, cy = make_K(d)
        cpos0 = d["cam_pos"][0]
        self.u[8:16] = np.array([fxy, cx, cy, np.float32(0.30),
                                 cpos0[0], cpos0[1], cpos0[2], 0.0],
                                np.float32).view(np.uint32)
        self.b_params = dev.create_buffer(
            size=64, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        q.write_buffer(self.b_params, 0, memoryview(self.u))

        self.bufs = [self.b_params, self.b_xyz, self.b_rot, self.b_scl,
                     self.b_opa, self.b_sh, self.b_slots, self.b_links,
                     self.b_vm, self.b_xyz_c, self.b_rot_c, self.b_means2d,
                     self.b_depths, self.b_conics, self.b_radii, self.b_colors,
                     self.b_opacity_b, self.b_ranges, self.b_counts,
                     self.b_offsets, self.b_cursors, self.b_keys_a,
                     self.b_flat_a, self.b_keys_b, self.b_flat_b,
                     self.b_hist2d, self.b_starts, self.b_rgb, self.b_depth,
                     self.b_renders, self.b_ranks, self.b_coltot,
                     self.b_bin_base]
        atomic = {18, 19, 20, 25, 27, 28, 31}
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
        self.module = dev.create_shader_module(code=WGSL)
        pl = dev.create_pipeline_layout(bind_group_layouts=[self.bgl])

        def pipe(entry):
            return dev.create_compute_pipeline(
                layout=pl, compute={"module": self.module, "entry_point": entry})

        names = ["update_links", "self_cull", "project", "isect_count",
                 "zero_counts", "scan_counts", "isect_fill",
                 "tile_starts", "rasterize", "quantize"]
        for p in range(NPASS):
            for step in ("zero_hist2d", "rank", "colscan", "binscan",
                         "scatter"):
                names.append(f"{step}_{p}")
        self.pipes = {name: pipe(name) for name in names}

        self.wg = {
            "update_links": (self.N + 255) // 256,
            "self_cull": (self.robot_n + 255) // 256,
            "project": (self.CN + 255) // 256,
            "isect_count": (self.CN + 255) // 256,
            "zero_counts": (self.C * self.nt + 1 + 255) // 256,
            "scan_counts": 1,
            "isect_fill": (self.CN + 255) // 256,
            "tile_starts": (self.C * self.nt + 1 + 255) // 256,
            "rasterize": self.C * self.nt,
            "quantize": (self.C * self.H * self.W + 255) // 256,
        }
        for p in range(NPASS):
            self.wg[f"zero_hist2d_{p}"] = (N_CHUNKS * 256 + 255) // 256
            self.wg[f"rank_{p}"] = N_CHUNKS
            self.wg[f"colscan_{p}"] = 256
            self.wg[f"binscan_{p}"] = 1
            self.wg[f"scatter_{p}"] = (CAP + 255) // 256
        self.order = (["update_links", "self_cull", "project", "zero_counts",
                       "isect_count", "scan_counts", "isect_fill"]
                      + [f"{s}_{p}" for p in range(NPASS)
                         for s in ("zero_hist2d", "rank", "colscan",
                                   "binscan", "scatter")]
                      + ["tile_starts", "rasterize", "quantize"])

    def set_links(self, pos, quat):
        """pos [13,3], quat [13,4] wxyz -> staging write."""
        links = np.zeros((13, 8), np.float32)
        links[:, :3] = pos
        links[:, 3:7] = quat
        self.queue.write_buffer(self.b_links, 0, memoryview(links.reshape(-1)))

    def render_frame(self):
        """One encoder, one submit; returns packed rgb/depth numpy arrays."""
        enc = self.device.create_command_encoder()
        cp = enc.begin_compute_pass()
        for name in self.order:
            cp.set_pipeline(self.pipes[name])
            cp.set_bind_group(0, self.bg)
            cp.dispatch_workgroups(self.wg[name])
        cp.end()
        self.queue.submit([enc.finish()])
        rgb = np.frombuffer(self.queue.read_buffer(self.b_rgb).cast("I"), np.uint32)
        depth = np.frombuffer(self.queue.read_buffer(self.b_depth).cast("I"), np.uint32)
        return rgb.reshape(self.C, self.H, self.W), depth.reshape(self.C, self.H, self.W)

    def unpack(self, rgb_packed, depth_mm):
        r = (rgb_packed & np.uint32(0xFF)).astype(np.uint8)
        g = ((rgb_packed >> np.uint32(8)) & np.uint32(0xFF)).astype(np.uint8)
        b = ((rgb_packed >> np.uint32(16)) & np.uint32(0xFF)).astype(np.uint8)
        rgb = np.stack([r, g, b], axis=-1)
        return rgb, depth_mm.astype(np.uint16)


