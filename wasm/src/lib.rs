#![no_std]

use core::cell::UnsafeCell;

#[cfg(target_arch = "wasm32")]
use core::panic::PanicInfo;

#[cfg(target_arch = "wasm32")]
#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use core::arch::wasm32::*;

const MAX_ENTITIES: usize = 65536;
const GX: usize = 32;
const GY: usize = 16;
const GZ: usize = 32;
const GRID_CELLS: usize = GX * GY * GZ; // 16384

const GRID_MIN_X: f32 = -72.0;
const GRID_MIN_Y: f32 = -36.0;
const GRID_MIN_Z: f32 = -72.0;
const CELL_SIZE: f32 = 4.5;
const INV_CELL_SIZE: f32 = 1.0 / CELL_SIZE;

const NEIGHBOR_RADIUS: f32 = 4.5;
const NEIGHBOR_RADIUS_SQ: f32 = NEIGHBOR_RADIUS * NEIGHBOR_RADIUS;
const SEP_RADIUS: f32 = 2.0;
const SEP_RADIUS_SQ: f32 = SEP_RADIUS * SEP_RADIUS;
const MAX_CELL_CHECKS: usize = 16;

const MIN_SPEED: f32 = 9.0;
const MAX_SPEED: f32 = 22.0;

// 6 outer architectural pillars at radius 38, angle k * PI / 3
const PILLAR_X: [f32; 6] = [
    38.0,
    19.0,
    -19.0,
    -38.0,
    -19.0,
    19.0,
];
const PILLAR_Z: [f32; 6] = [
    0.0,
    32.908965,
    32.908965,
    0.0,
    -32.908965,
    -32.908965,
];
const PILLAR_AVOID_R: f32 = 5.6;
const OBELISK_AVOID_R: f32 = 8.2;

#[repr(align(16))]
struct EngineState {
    // Active Boid SoA state (16-byte aligned for SIMD128)
    px: [f32; MAX_ENTITIES],
    py: [f32; MAX_ENTITIES],
    pz: [f32; MAX_ENTITIES],
    vx: [f32; MAX_ENTITIES],
    vy: [f32; MAX_ENTITIES],
    vz: [f32; MAX_ENTITIES],
    phase: [f32; MAX_ENTITIES],
    color_seed: [f32; MAX_ENTITIES],

    // Spatially-sorted SoA double-buffer (contiguous memory for SIMD neighbor scans)
    spx: [f32; MAX_ENTITIES],
    spy: [f32; MAX_ENTITIES],
    spz: [f32; MAX_ENTITIES],
    svx: [f32; MAX_ENTITIES],
    svy: [f32; MAX_ENTITIES],
    svz: [f32; MAX_ENTITIES],
    sphase: [f32; MAX_ENTITIES],
    scolor: [f32; MAX_ENTITIES],

    // Spatial Hash Grid buffers
    cell_id: [u32; MAX_ENTITIES],
    cell_count: [u32; GRID_CELLS],
    cell_start: [u32; GRID_CELLS + 1],
    cell_cursor: [u32; GRID_CELLS],

    // Packed GPU instance output buffer: 8 f32 (32 bytes) per instance
    gpu_out: [f32; MAX_ENTITIES * 8],

    // 6 frustum planes * 4 floats (nx, ny, nz, d)
    frustum: [f32; 24],

    // Scenario 3: Crystals static orbital parameters (SoA, SIMD128 aligned)
    cr_radius: [f32; MAX_ENTITIES],
    cr_angle: [f32; MAX_ENTITIES],
    cr_base_y: [f32; MAX_ENTITIES],
    cr_orbit_spd: [f32; MAX_ENTITIES],
    cr_spin_spd: [f32; MAX_ENTITIES],
    cr_scale: [f32; MAX_ENTITIES],
    cr_phase: [f32; MAX_ENTITIES],
    cr_color: [f32; MAX_ENTITIES],
}

struct SyncState(UnsafeCell<EngineState>);
// Safe in single-threaded wasm32 execution model
unsafe impl Sync for SyncState {}

static STATE: SyncState = SyncState(UnsafeCell::new(EngineState {
    px: [0.0; MAX_ENTITIES],
    py: [0.0; MAX_ENTITIES],
    pz: [0.0; MAX_ENTITIES],
    vx: [0.0; MAX_ENTITIES],
    vy: [0.0; MAX_ENTITIES],
    vz: [0.0; MAX_ENTITIES],
    phase: [0.0; MAX_ENTITIES],
    color_seed: [0.0; MAX_ENTITIES],
    spx: [0.0; MAX_ENTITIES],
    spy: [0.0; MAX_ENTITIES],
    spz: [0.0; MAX_ENTITIES],
    svx: [0.0; MAX_ENTITIES],
    svy: [0.0; MAX_ENTITIES],
    svz: [0.0; MAX_ENTITIES],
    sphase: [0.0; MAX_ENTITIES],
    scolor: [0.0; MAX_ENTITIES],
    cell_id: [0; MAX_ENTITIES],
    cell_count: [0; GRID_CELLS],
    cell_start: [0; GRID_CELLS + 1],
    cell_cursor: [0; GRID_CELLS],
    gpu_out: [0.0; MAX_ENTITIES * 8],
    frustum: [0.0; 24],
    cr_radius: [0.0; MAX_ENTITIES],
    cr_angle: [0.0; MAX_ENTITIES],
    cr_base_y: [0.0; MAX_ENTITIES],
    cr_orbit_spd: [0.0; MAX_ENTITIES],
    cr_spin_spd: [0.0; MAX_ENTITIES],
    cr_scale: [0.0; MAX_ENTITIES],
    cr_phase: [0.0; MAX_ENTITIES],
    cr_color: [0.0; MAX_ENTITIES],
}));

#[inline(always)]
unsafe fn state_mut() -> &'static mut EngineState {
    &mut *STATE.0.get()
}

// --- Pointer getters for zero-copy JS typed array views ---

#[no_mangle]
pub extern "C" fn is_simd_enabled() -> u32 {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        1
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        0
    }
}

#[no_mangle] pub extern "C" fn get_px_ptr() -> *mut f32 { unsafe { state_mut().px.as_mut_ptr() } }
#[no_mangle] pub extern "C" fn get_py_ptr() -> *mut f32 { unsafe { state_mut().py.as_mut_ptr() } }
#[no_mangle] pub extern "C" fn get_pz_ptr() -> *mut f32 { unsafe { state_mut().pz.as_mut_ptr() } }
#[no_mangle] pub extern "C" fn get_vx_ptr() -> *mut f32 { unsafe { state_mut().vx.as_mut_ptr() } }
#[no_mangle] pub extern "C" fn get_vy_ptr() -> *mut f32 { unsafe { state_mut().vy.as_mut_ptr() } }
#[no_mangle] pub extern "C" fn get_vz_ptr() -> *mut f32 { unsafe { state_mut().vz.as_mut_ptr() } }
#[no_mangle] pub extern "C" fn get_phase_ptr() -> *mut f32 { unsafe { state_mut().phase.as_mut_ptr() } }
#[no_mangle] pub extern "C" fn get_color_ptr() -> *mut f32 { unsafe { state_mut().color_seed.as_mut_ptr() } }
#[no_mangle] pub extern "C" fn get_gpu_out_ptr() -> *mut f32 { unsafe { state_mut().gpu_out.as_mut_ptr() } }
#[no_mangle] pub extern "C" fn get_frustum_ptr() -> *mut f32 { unsafe { state_mut().frustum.as_mut_ptr() } }

#[no_mangle] pub extern "C" fn get_cr_radius_ptr() -> *mut f32 { unsafe { state_mut().cr_radius.as_mut_ptr() } }
#[no_mangle] pub extern "C" fn get_cr_angle_ptr() -> *mut f32 { unsafe { state_mut().cr_angle.as_mut_ptr() } }
#[no_mangle] pub extern "C" fn get_cr_base_y_ptr() -> *mut f32 { unsafe { state_mut().cr_base_y.as_mut_ptr() } }
#[no_mangle] pub extern "C" fn get_cr_orbit_spd_ptr() -> *mut f32 { unsafe { state_mut().cr_orbit_spd.as_mut_ptr() } }
#[no_mangle] pub extern "C" fn get_cr_spin_spd_ptr() -> *mut f32 { unsafe { state_mut().cr_spin_spd.as_mut_ptr() } }
#[no_mangle] pub extern "C" fn get_cr_scale_ptr() -> *mut f32 { unsafe { state_mut().cr_scale.as_mut_ptr() } }
#[no_mangle] pub extern "C" fn get_cr_phase_ptr() -> *mut f32 { unsafe { state_mut().cr_phase.as_mut_ptr() } }
#[no_mangle] pub extern "C" fn get_cr_color_ptr() -> *mut f32 { unsafe { state_mut().cr_color.as_mut_ptr() } }

#[inline(always)]
fn fast_sqrt(x: f32) -> f32 {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        f32x4_extract_lane::<0>(f32x4_sqrt(f32x4_splat(x)))
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        if x <= 0.0 {
            return 0.0;
        }
        let mut y = f32::from_bits((x.to_bits() >> 1) + 0x1fc0_0000);
        y = 0.5 * (y + x / y);
        y = 0.5 * (y + x / y);
        y = 0.5 * (y + x / y);
        y
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
unsafe fn hsum_f32x4(v: v128) -> f32 {
    let shuf = i32x4_shuffle::<2, 3, 0, 1>(v, v);
    let sum2 = f32x4_add(v, shuf);
    let shuf2 = i32x4_shuffle::<1, 0, 3, 2>(sum2, sum2);
    f32x4_extract_lane::<0>(f32x4_add(sum2, shuf2))
}

#[inline(always)]
fn clamp_i32(v: i32, lo: i32, hi: i32) -> i32 {
    if v < lo { lo } else if v > hi { hi } else { v }
}

#[inline(always)]
fn cell_index(cx: i32, cy: i32, cz: i32) -> usize {
    ((cy as usize) * GZ + (cz as usize)) * GX + (cx as usize)
}

/// Scenario 1: Spatial-Hash 3D Boids + Pillar Collisions + Pointer Vortex + SIMD128 Neighbor Scan & Frustum Culling
#[no_mangle]
pub extern "C" fn step_boids(
    count: u32,
    dt: f32,
    ptr_x: f32,
    ptr_y: f32,
    ptr_z: f32,
    ptr_mode: i32,
    sep_weight: f32,
    ali_weight: f32,
    coh_weight: f32,
    do_cull: u32,
) -> u32 {
    let n = (count as usize).min(MAX_ENTITIES);
    if n == 0 {
        return 0;
    }
    let dt = if dt > 0.05 { 0.05 } else if dt < 0.0001 { 0.0001 } else { dt };

    unsafe {
        let s = state_mut();

        // 1. Zero grid counts
        core::ptr::write_bytes(s.cell_count.as_mut_ptr(), 0, GRID_CELLS);

        // 2. Hash each bird into 3D uniform grid cell
        for i in 0..n {
            let cx = clamp_i32(((s.px[i] - GRID_MIN_X) * INV_CELL_SIZE) as i32, 0, (GX - 1) as i32);
            let cy = clamp_i32(((s.py[i] - GRID_MIN_Y) * INV_CELL_SIZE) as i32, 0, (GY - 1) as i32);
            let cz = clamp_i32(((s.pz[i] - GRID_MIN_Z) * INV_CELL_SIZE) as i32, 0, (GZ - 1) as i32);
            let cid = cell_index(cx, cy, cz);
            s.cell_id[i] = cid as u32;
            s.cell_count[cid] += 1;
        }

        // 3. Prefix sum for cell start offsets
        let mut sum: u32 = 0;
        for c in 0..GRID_CELLS {
            s.cell_start[c] = sum;
            s.cell_cursor[c] = sum;
            sum += s.cell_count[c];
        }
        s.cell_start[GRID_CELLS] = sum;

        // 4. Counting-sort scatter into spatially coherent SoA arrays
        for i in 0..n {
            let cid = s.cell_id[i] as usize;
            let dst = s.cell_cursor[cid] as usize;
            s.cell_cursor[cid] = (dst + 1) as u32;
            s.spx[dst] = s.px[i];
            s.spy[dst] = s.py[i];
            s.spz[dst] = s.pz[i];
            s.svx[dst] = s.vx[i];
            s.svy[dst] = s.vy[i];
            s.svz[dst] = s.vz[i];
            s.sphase[dst] = s.phase[i];
            s.scolor[dst] = s.color_seed[i];
        }

        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        let (eps_v, rad_sq_v, sep_sq_v, sep_eps_v, one_v, zero_v) = (
            f32x4_splat(0.0001),
            f32x4_splat(NEIGHBOR_RADIUS_SQ),
            f32x4_splat(SEP_RADIUS_SQ),
            f32x4_splat(0.15),
            f32x4_splat(1.0),
            f32x4_splat(0.0),
        );

        let spx_ptr = s.spx.as_ptr();
        let spy_ptr = s.spy.as_ptr();
        let spz_ptr = s.spz.as_ptr();
        let svx_ptr = s.svx.as_ptr();
        let svy_ptr = s.svy.as_ptr();
        let svz_ptr = s.svz.as_ptr();

        // 5. Evaluate 3D spatial-hash flocking + obstacles + bounds + integrate
        for i in 0..n {
            let x = *spx_ptr.add(i);
            let y = *spy_ptr.add(i);
            let z = *spz_ptr.add(i);
            let mut bvx = *svx_ptr.add(i);
            let mut bvy = *svy_ptr.add(i);
            let mut bvz = *svz_ptr.add(i);

            let cx = clamp_i32(((x - GRID_MIN_X) * INV_CELL_SIZE) as i32, 0, (GX - 1) as i32);
            let cy = clamp_i32(((y - GRID_MIN_Y) * INV_CELL_SIZE) as i32, 0, (GY - 1) as i32);
            let cz = clamp_i32(((z - GRID_MIN_Z) * INV_CELL_SIZE) as i32, 0, (GZ - 1) as i32);

            let min_x = if cx > 0 { cx - 1 } else { 0 };
            let max_x = if cx + 1 < GX as i32 { cx + 1 } else { (GX - 1) as i32 };
            let min_y = if cy > 0 { cy - 1 } else { 0 };
            let max_y = if cy + 1 < GY as i32 { cy + 1 } else { (GY - 1) as i32 };
            let min_z = if cz > 0 { cz - 1 } else { 0 };
            let max_z = if cz + 1 < GZ as i32 { cz + 1 } else { (GZ - 1) as i32 };

            let mut sep_x = 0.0f32;
            let mut sep_y = 0.0f32;
            let mut sep_z = 0.0f32;
            let mut ali_x = 0.0f32;
            let mut ali_y = 0.0f32;
            let mut ali_z = 0.0f32;
            let mut coh_x = 0.0f32;
            let mut coh_y = 0.0f32;
            let mut coh_z = 0.0f32;
            let mut neighbors: f32 = 0.0;

            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            let (
                xv, yv, zv,
                mut v_sep_x, mut v_sep_y, mut v_sep_z,
                mut v_ali_x, mut v_ali_y, mut v_ali_z,
                mut v_coh_x, mut v_coh_y, mut v_coh_z,
                mut v_cnt,
            ) = (
                f32x4_splat(x), f32x4_splat(y), f32x4_splat(z),
                zero_v, zero_v, zero_v,
                zero_v, zero_v, zero_v,
                zero_v, zero_v, zero_v,
                zero_v,
            );

            let mut ny = min_y;
            while ny <= max_y {
                let mut nz = min_z;
                while nz <= max_z {
                    let start_cell = cell_index(min_x, ny, nz);
                    let end_cell = cell_index(max_x, ny, nz);
                    let j_start = s.cell_start[start_cell] as usize;
                    let mut j_end = s.cell_start[end_cell + 1] as usize;
                    if j_end - j_start > MAX_CELL_CHECKS * 3 {
                        j_end = j_start + MAX_CELL_CHECKS * 3;
                    }

                    let mut j = j_start;

                    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                    {
                        while j + 4 <= j_end {
                            let jx = v128_load(spx_ptr.add(j) as *const v128);
                            let jy = v128_load(spy_ptr.add(j) as *const v128);
                            let jz = v128_load(spz_ptr.add(j) as *const v128);

                            let dx = f32x4_sub(xv, jx);
                            let dy = f32x4_sub(yv, jy);
                            let dz = f32x4_sub(zv, jz);

                            let d2 = f32x4_add(
                                f32x4_add(f32x4_mul(dx, dx), f32x4_mul(dy, dy)),
                                f32x4_mul(dz, dz),
                            );

                            let in_range = v128_and(f32x4_gt(d2, eps_v), f32x4_lt(d2, rad_sq_v));
                            if v128_any_true(in_range) {
                                let jvx = v128_load(svx_ptr.add(j) as *const v128);
                                let jvy = v128_load(svy_ptr.add(j) as *const v128);
                                let jvz = v128_load(svz_ptr.add(j) as *const v128);

                                v_cnt = f32x4_add(v_cnt, v128_and(one_v, in_range));
                                v_ali_x = f32x4_add(v_ali_x, v128_and(jvx, in_range));
                                v_ali_y = f32x4_add(v_ali_y, v128_and(jvy, in_range));
                                v_ali_z = f32x4_add(v_ali_z, v128_and(jvz, in_range));
                                v_coh_x = f32x4_add(v_coh_x, v128_and(jx, in_range));
                                v_coh_y = f32x4_add(v_coh_y, v128_and(jy, in_range));
                                v_coh_z = f32x4_add(v_coh_z, v128_and(jz, in_range));

                                let in_sep = v128_and(in_range, f32x4_lt(d2, sep_sq_v));
                                if v128_any_true(in_sep) {
                                    let inv = v128_and(f32x4_div(one_v, f32x4_add(d2, sep_eps_v)), in_sep);
                                    v_sep_x = f32x4_add(v_sep_x, f32x4_mul(dx, inv));
                                    v_sep_y = f32x4_add(v_sep_y, f32x4_mul(dy, inv));
                                    v_sep_z = f32x4_add(v_sep_z, f32x4_mul(dz, inv));
                                }
                            }
                            j += 4;
                        }
                    }

                    while j < j_end {
                        let jx = *spx_ptr.add(j);
                        let jy = *spy_ptr.add(j);
                        let jz = *spz_ptr.add(j);
                        let dx = x - jx;
                        let dy = y - jy;
                        let dz = z - jz;
                        let d2 = dx * dx + dy * dy + dz * dz;
                        if d2 > 0.0001 && d2 < NEIGHBOR_RADIUS_SQ {
                            neighbors += 1.0;
                            ali_x += *svx_ptr.add(j);
                            ali_y += *svy_ptr.add(j);
                            ali_z += *svz_ptr.add(j);
                            coh_x += jx;
                            coh_y += jy;
                            coh_z += jz;
                            if d2 < SEP_RADIUS_SQ {
                                let inv = 1.0 / (d2 + 0.15);
                                sep_x += dx * inv;
                                sep_y += dy * inv;
                                sep_z += dz * inv;
                            }
                        }
                        j += 1;
                    }
                    nz += 1;
                }
                ny += 1;
            }

            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            {
                neighbors += hsum_f32x4(v_cnt);
                sep_x += hsum_f32x4(v_sep_x);
                sep_y += hsum_f32x4(v_sep_y);
                sep_z += hsum_f32x4(v_sep_z);
                ali_x += hsum_f32x4(v_ali_x);
                ali_y += hsum_f32x4(v_ali_y);
                ali_z += hsum_f32x4(v_ali_z);
                coh_x += hsum_f32x4(v_coh_x);
                coh_y += hsum_f32x4(v_coh_y);
                coh_z += hsum_f32x4(v_coh_z);
            }

            let mut ax = 0.0f32;
            let mut ay = 0.0f32;
            let mut az = 0.0f32;

            if neighbors > 0.0 {
                let inv_n = 1.0 / neighbors;
                ax += sep_x * sep_weight;
                ay += sep_y * sep_weight;
                az += sep_z * sep_weight;

                ax += (ali_x * inv_n - bvx) * ali_weight;
                ay += (ali_y * inv_n - bvy) * ali_weight;
                az += (ali_z * inv_n - bvz) * ali_weight;

                ax += (coh_x * inv_n - x) * coh_weight;
                ay += (coh_y * inv_n - y) * coh_weight;
                az += (coh_z * inv_n - z) * coh_weight;
            }

            // Gentle swirling sanctuary current + central obelisk + outer containment
            let r_xz_sq = x * x + z * z;
            if r_xz_sq > 0.01 {
                let r_xz = fast_sqrt(r_xz_sq);
                let inv_r = 1.0 / r_xz;
                ax += -z * inv_r * 2.2;
                az += x * inv_r * 2.2;

                if r_xz < OBELISK_AVOID_R && y < 24.0 {
                    let push = (OBELISK_AVOID_R - r_xz) * 14.0;
                    ax += x * inv_r * push;
                    az += z * inv_r * push;
                }

                if r_xz > 52.0 {
                    let over = r_xz - 52.0;
                    ax -= x * inv_r * over * 3.2;
                    az -= z * inv_r * over * 3.2;
                }
            }

            // Avoid 6 surrounding architectural monolith pillars
            for p in 0..6 {
                let dx = x - PILLAR_X[p];
                let dz = z - PILLAR_Z[p];
                let d2 = dx * dx + dz * dz;
                if d2 < PILLAR_AVOID_R * PILLAR_AVOID_R && d2 > 0.0001 && y < 26.0 {
                    let d = fast_sqrt(d2);
                    let push = (PILLAR_AVOID_R - d) * 16.0 / d;
                    ax += dx * push;
                    az += dz * push;
                }
            }

            // Vertical floor & ceiling gentle cushion
            if y > 24.0 {
                ay -= (y - 24.0) * 4.5;
            } else if y < -24.0 {
                ay += (-24.0 - y) * 5.0;
            }
            ay += (2.0 - y) * 0.16;

            // Interactive 3D Pointer Vortex
            if ptr_mode != 0 {
                let dx = ptr_x - x;
                let dy = ptr_y - y;
                let dz = ptr_z - z;
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 < 28.0 * 28.0 && d2 > 0.01 {
                    let d = fast_sqrt(d2);
                    let falloff = 1.0 - d / 28.0;
                    let force = (ptr_mode as f32) * falloff * 52.0 / d;
                    ax += dx * force;
                    ay += dy * force;
                    az += dz * force;
                }
            }

            bvx += ax * dt;
            bvy += ay * dt;
            bvz += az * dt;

            let spd = fast_sqrt(bvx * bvx + bvy * bvy + bvz * bvz);
            if spd > MAX_SPEED {
                let inv_s = MAX_SPEED / spd;
                bvx *= inv_s;
                bvy *= inv_s;
                bvz *= inv_s;
            } else if spd < MIN_SPEED {
                if spd > 0.0001 {
                    let inv_s = MIN_SPEED / spd;
                    bvx *= inv_s;
                    bvy *= inv_s;
                    bvz *= inv_s;
                } else {
                    bvx = MIN_SPEED;
                }
            }

            let nx = x + bvx * dt;
            let ny_pos = y + bvy * dt;
            let nz_pos = z + bvz * dt;

            let climb_boost = if bvy > 0.0 { bvy * 0.28 } else { 0.0 };
            let mut nphase = s.sphase[i] + dt * (9.5 + spd * 0.25 + climb_boost);
            if nphase > 62.831853 {
                nphase -= 62.831853;
            }

            s.px[i] = nx;
            s.py[i] = ny_pos;
            s.pz[i] = nz_pos;
            s.vx[i] = bvx;
            s.vy[i] = bvy;
            s.vz[i] = bvz;
            s.phase[i] = nphase;
            s.color_seed[i] = s.scolor[i];
        }

        // 6. SIMD128 4-wide Frustum Culling + Compact 32-byte GPU Buffer Packing
        pack_boids_gpu(s, n, do_cull != 0)
    }
}

#[inline(always)]
unsafe fn pack_boids_gpu(s: &mut EngineState, n: usize, do_cull: bool) -> u32 {
    let out = &mut s.gpu_out;
    let fr = &s.frustum;

    if !do_cull {
        for i in 0..n {
            let o = i * 8;
            out[o] = s.px[i];
            out[o + 1] = s.py[i];
            out[o + 2] = s.pz[i];
            out[o + 3] = s.phase[i];
            out[o + 4] = s.vx[i];
            out[o + 5] = s.vy[i];
            out[o + 6] = s.vz[i];
            out[o + 7] = s.color_seed[i];
        }
        return n as u32;
    }

    let bound_r = 2.6f32;
    let mut vis: usize = 0;
    let mut i = 0usize;

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        let p0x = f32x4_splat(fr[0]);  let p0y = f32x4_splat(fr[1]);  let p0z = f32x4_splat(fr[2]);  let p0d = f32x4_splat(fr[3] + bound_r);
        let p1x = f32x4_splat(fr[4]);  let p1y = f32x4_splat(fr[5]);  let p1z = f32x4_splat(fr[6]);  let p1d = f32x4_splat(fr[7] + bound_r);
        let p2x = f32x4_splat(fr[8]);  let p2y = f32x4_splat(fr[9]);  let p2z = f32x4_splat(fr[10]); let p2d = f32x4_splat(fr[11] + bound_r);
        let p3x = f32x4_splat(fr[12]); let p3y = f32x4_splat(fr[13]); let p3z = f32x4_splat(fr[14]); let p3d = f32x4_splat(fr[15] + bound_r);
        let p4x = f32x4_splat(fr[16]); let p4y = f32x4_splat(fr[17]); let p4z = f32x4_splat(fr[18]); let p4d = f32x4_splat(fr[19] + bound_r);
        let p5x = f32x4_splat(fr[20]); let p5y = f32x4_splat(fr[21]); let p5z = f32x4_splat(fr[22]); let p5d = f32x4_splat(fr[23] + bound_r);
        let zero = f32x4_splat(0.0);

        while i + 4 <= n {
            let x4 = v128_load(s.px.as_ptr().add(i) as *const v128);
            let y4 = v128_load(s.py.as_ptr().add(i) as *const v128);
            let z4 = v128_load(s.pz.as_ptr().add(i) as *const v128);

            let d0 = f32x4_add(f32x4_add(f32x4_mul(x4, p0x), f32x4_mul(y4, p0y)), f32x4_add(f32x4_mul(z4, p0z), p0d));
            let d1 = f32x4_add(f32x4_add(f32x4_mul(x4, p1x), f32x4_mul(y4, p1y)), f32x4_add(f32x4_mul(z4, p1z), p1d));
            let d2 = f32x4_add(f32x4_add(f32x4_mul(x4, p2x), f32x4_mul(y4, p2y)), f32x4_add(f32x4_mul(z4, p2z), p2d));
            let d3 = f32x4_add(f32x4_add(f32x4_mul(x4, p3x), f32x4_mul(y4, p3y)), f32x4_add(f32x4_mul(z4, p3z), p3d));
            let d4 = f32x4_add(f32x4_add(f32x4_mul(x4, p4x), f32x4_mul(y4, p4y)), f32x4_add(f32x4_mul(z4, p4z), p4d));
            let d5 = f32x4_add(f32x4_add(f32x4_mul(x4, p5x), f32x4_mul(y4, p5y)), f32x4_add(f32x4_mul(z4, p5z), p5d));

            let m = v128_and(
                v128_and(v128_and(f32x4_ge(d0, zero), f32x4_ge(d1, zero)), f32x4_ge(d2, zero)),
                v128_and(v128_and(f32x4_ge(d3, zero), f32x4_ge(d4, zero)), f32x4_ge(d5, zero)),
            );
            let mask = i32x4_bitmask(m);

            if mask != 0 {
                for lane in 0..4 {
                    if (mask & (1 << lane)) != 0 {
                        let idx = i + lane;
                        let o = vis * 8;
                        out[o] = s.px[idx];
                        out[o + 1] = s.py[idx];
                        out[o + 2] = s.pz[idx];
                        out[o + 3] = s.phase[idx];
                        out[o + 4] = s.vx[idx];
                        out[o + 5] = s.vy[idx];
                        out[o + 6] = s.vz[idx];
                        out[o + 7] = s.color_seed[idx];
                        vis += 1;
                    }
                }
            }
            i += 4;
        }
    }

    while i < n {
        let x = s.px[i];
        let y = s.py[i];
        let z = s.pz[i];
        if (x * fr[0] + y * fr[1] + z * fr[2] + fr[3] + bound_r) >= 0.0
            && (x * fr[4] + y * fr[5] + z * fr[6] + fr[7] + bound_r) >= 0.0
            && (x * fr[8] + y * fr[9] + z * fr[10] + fr[11] + bound_r) >= 0.0
            && (x * fr[12] + y * fr[13] + z * fr[14] + fr[15] + bound_r) >= 0.0
            && (x * fr[16] + y * fr[17] + z * fr[18] + fr[19] + bound_r) >= 0.0
            && (x * fr[20] + y * fr[21] + z * fr[22] + fr[23] + bound_r) >= 0.0
        {
            let o = vis * 8;
            out[o] = x;
            out[o + 1] = y;
            out[o + 2] = z;
            out[o + 3] = s.phase[i];
            out[o + 4] = s.vx[i];
            out[o + 5] = s.vy[i];
            out[o + 6] = s.vz[i];
            out[o + 7] = s.color_seed[i];
            vis += 1;
        }
        i += 1;
    }

    vis as u32
}

// Fast accurate sin/cos in [-PI, PI] for Scenario 3 (SIMD128 + scalar)
const PI: f32 = 3.1415926535897932;
const TAU: f32 = 6.2831853071795865;
const INV_TAU: f32 = 1.0 / TAU;

#[inline(always)]
fn scalar_sin_cos(angle: f32) -> (f32, f32) {
    let mut x = angle - ((angle * INV_TAU + 0.5) as i32 as f32) * TAU;
    if x > PI { x -= TAU; }
    if x < -PI { x += TAU; }
    let (sx, sign_c) = if x > 1.5707963 {
        (PI - x, -1.0f32)
    } else if x < -1.5707963 {
        (-PI - x, -1.0f32)
    } else {
        (x, 1.0f32)
    };
    let x2 = sx * sx;
    let s = sx * (1.0 + x2 * (-0.16666667 + x2 * (0.008333333 + x2 * -0.0001984127)));
    let c = sign_c * (1.0 + x2 * (-0.5 + x2 * (0.041666668 + x2 * -0.0013888889)));
    (s, c)
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
unsafe fn simd_sin_cos(angle: v128) -> (v128, v128) {
    let inv_tau = f32x4_splat(INV_TAU);
    let tau = f32x4_splat(TAU);
    let pi = f32x4_splat(PI);
    let half_pi = f32x4_splat(1.5707963);
    let neg_half_pi = f32x4_splat(-1.5707963);

    let k = f32x4_nearest(f32x4_mul(angle, inv_tau));
    let x = f32x4_sub(angle, f32x4_mul(k, tau));

    let gt = f32x4_gt(x, half_pi);
    let lt = f32x4_lt(x, neg_half_pi);
    let sx_gt = f32x4_sub(pi, x);
    let sx_lt = f32x4_sub(f32x4_neg(pi), x);
    let sx = v128_bitselect(sx_gt, v128_bitselect(sx_lt, x, lt), gt);

    let flipped = v128_or(gt, lt);
    let sign_c = v128_bitselect(f32x4_splat(-1.0), f32x4_splat(1.0), flipped);

    let x2 = f32x4_mul(sx, sx);

    let mut sp = f32x4_splat(-0.0001984127);
    sp = f32x4_add(f32x4_mul(sp, x2), f32x4_splat(0.008333333));
    sp = f32x4_add(f32x4_mul(sp, x2), f32x4_splat(-0.16666667));
    sp = f32x4_add(f32x4_mul(sp, x2), f32x4_splat(1.0));
    let s = f32x4_mul(sx, sp);

    let mut cp = f32x4_splat(-0.0013888889);
    cp = f32x4_add(f32x4_mul(cp, x2), f32x4_splat(0.041666668));
    cp = f32x4_add(f32x4_mul(cp, x2), f32x4_splat(-0.5));
    cp = f32x4_add(f32x4_mul(cp, x2), f32x4_splat(1.0));
    let c = f32x4_mul(sign_c, cp);

    (s, c)
}

/// Scenario 3: SIMD128 4-wide update of N individual 3D Crystal Monolith transforms
#[no_mangle]
pub extern "C" fn step_crystals(count: u32, time: f32, do_cull: u32) -> u32 {
    let n = (count as usize).min(MAX_ENTITIES);
    if n == 0 {
        return 0;
    }

    unsafe {
        let s = state_mut();
        let fr = &s.frustum;
        let out = &mut s.gpu_out;

        let mut vis: u32 = 0;
        let mut i = 0usize;
        let cull = do_cull != 0;
        let bound_r = 2.2f32;

        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            let t4 = f32x4_splat(time);
            let bob_freq = f32x4_splat(1.7);
            let bob_amp = f32x4_splat(2.4);
            let tilt_k = f32x4_splat(0.6);
            let half = f32x4_splat(0.5);

            let p0x = f32x4_splat(fr[0]);  let p0y = f32x4_splat(fr[1]);  let p0z = f32x4_splat(fr[2]);  let p0d = f32x4_splat(fr[3] + bound_r);
            let p1x = f32x4_splat(fr[4]);  let p1y = f32x4_splat(fr[5]);  let p1z = f32x4_splat(fr[6]);  let p1d = f32x4_splat(fr[7] + bound_r);
            let p2x = f32x4_splat(fr[8]);  let p2y = f32x4_splat(fr[9]);  let p2z = f32x4_splat(fr[10]); let p2d = f32x4_splat(fr[11] + bound_r);
            let p3x = f32x4_splat(fr[12]); let p3y = f32x4_splat(fr[13]); let p3z = f32x4_splat(fr[14]); let p3d = f32x4_splat(fr[15] + bound_r);
            let p4x = f32x4_splat(fr[16]); let p4y = f32x4_splat(fr[17]); let p4z = f32x4_splat(fr[18]); let p4d = f32x4_splat(fr[19] + bound_r);
            let p5x = f32x4_splat(fr[20]); let p5y = f32x4_splat(fr[21]); let p5z = f32x4_splat(fr[22]); let p5d = f32x4_splat(fr[23] + bound_r);
            let zero = f32x4_splat(0.0);

            while i + 4 <= n {
                let r4 = v128_load(s.cr_radius.as_ptr().add(i) as *const v128);
                let a0 = v128_load(s.cr_angle.as_ptr().add(i) as *const v128);
                let by = v128_load(s.cr_base_y.as_ptr().add(i) as *const v128);
                let ospd = v128_load(s.cr_orbit_spd.as_ptr().add(i) as *const v128);
                let sspd = v128_load(s.cr_spin_spd.as_ptr().add(i) as *const v128);
                let sc4 = v128_load(s.cr_scale.as_ptr().add(i) as *const v128);
                let ph4 = v128_load(s.cr_phase.as_ptr().add(i) as *const v128);
                let col4 = v128_load(s.cr_color.as_ptr().add(i) as *const v128);

                let ang = f32x4_add(a0, f32x4_mul(ospd, t4));
                let (sa, ca) = simd_sin_cos(ang);
                let x4 = f32x4_mul(ca, r4);
                let z4 = f32x4_mul(sa, r4);

                let bob_arg = f32x4_add(f32x4_mul(t4, bob_freq), ph4);
                let (sb, _) = simd_sin_cos(bob_arg);
                let y4 = f32x4_add(by, f32x4_mul(sb, bob_amp));
                let pulse4 = f32x4_add(f32x4_mul(sb, half), half);

                let ry4 = f32x4_add(ph4, f32x4_mul(sspd, t4));
                let rx4 = f32x4_mul(sb, tilt_k);

                let mask = if cull {
                    let d0 = f32x4_add(f32x4_add(f32x4_mul(x4, p0x), f32x4_mul(y4, p0y)), f32x4_add(f32x4_mul(z4, p0z), p0d));
                    let d1 = f32x4_add(f32x4_add(f32x4_mul(x4, p1x), f32x4_mul(y4, p1y)), f32x4_add(f32x4_mul(z4, p1z), p1d));
                    let d2 = f32x4_add(f32x4_add(f32x4_mul(x4, p2x), f32x4_mul(y4, p2y)), f32x4_add(f32x4_mul(z4, p2z), p2d));
                    let d3 = f32x4_add(f32x4_add(f32x4_mul(x4, p3x), f32x4_mul(y4, p3y)), f32x4_add(f32x4_mul(z4, p3z), p3d));
                    let d4 = f32x4_add(f32x4_add(f32x4_mul(x4, p4x), f32x4_mul(y4, p4y)), f32x4_add(f32x4_mul(z4, p4z), p4d));
                    let d5 = f32x4_add(f32x4_add(f32x4_mul(x4, p5x), f32x4_mul(y4, p5y)), f32x4_add(f32x4_mul(z4, p5z), p5d));
                    let m = v128_and(
                        v128_and(v128_and(f32x4_ge(d0, zero), f32x4_ge(d1, zero)), f32x4_ge(d2, zero)),
                        v128_and(v128_and(f32x4_ge(d3, zero), f32x4_ge(d4, zero)), f32x4_ge(d5, zero)),
                    );
                    i32x4_bitmask(m)
                } else {
                    0xF
                };

                let mut xv = [0.0f32; 4]; v128_store(xv.as_mut_ptr() as *mut v128, x4);
                let mut yv = [0.0f32; 4]; v128_store(yv.as_mut_ptr() as *mut v128, y4);
                let mut zv = [0.0f32; 4]; v128_store(zv.as_mut_ptr() as *mut v128, z4);
                let mut scv = [0.0f32; 4]; v128_store(scv.as_mut_ptr() as *mut v128, sc4);
                let mut ryv = [0.0f32; 4]; v128_store(ryv.as_mut_ptr() as *mut v128, ry4);
                let mut rxv = [0.0f32; 4]; v128_store(rxv.as_mut_ptr() as *mut v128, rx4);
                let mut cv = [0.0f32; 4]; v128_store(cv.as_mut_ptr() as *mut v128, col4);
                let mut pv = [0.0f32; 4]; v128_store(pv.as_mut_ptr() as *mut v128, pulse4);

                for lane in 0..4 {
                    let o = (i + lane) * 8;
                    let visible = (mask & (1 << lane)) != 0;
                    if visible { vis += 1; }
                    out[o] = xv[lane];
                    out[o + 1] = yv[lane];
                    out[o + 2] = zv[lane];
                    out[o + 3] = if visible { scv[lane] } else { 0.0 };
                    out[o + 4] = ryv[lane];
                    out[o + 5] = rxv[lane];
                    out[o + 6] = cv[lane];
                    out[o + 7] = pv[lane];
                }
                i += 4;
            }
        }

        while i < n {
            let ang = s.cr_angle[i] + s.cr_orbit_spd[i] * time;
            let (sa, ca) = scalar_sin_cos(ang);
            let x = ca * s.cr_radius[i];
            let z = sa * s.cr_radius[i];
            let (sb, _) = scalar_sin_cos(time * 1.7 + s.cr_phase[i]);
            let y = s.cr_base_y[i] + sb * 2.4;
            let ry = s.cr_phase[i] + s.cr_spin_spd[i] * time;
            let rx = sb * 0.6;
            let pulse = sb * 0.5 + 0.5;

            let visible = !cull || (
                (x * fr[0] + y * fr[1] + z * fr[2] + fr[3] + bound_r) >= 0.0
                && (x * fr[4] + y * fr[5] + z * fr[6] + fr[7] + bound_r) >= 0.0
                && (x * fr[8] + y * fr[9] + z * fr[10] + fr[11] + bound_r) >= 0.0
                && (x * fr[12] + y * fr[13] + z * fr[14] + fr[15] + bound_r) >= 0.0
                && (x * fr[16] + y * fr[17] + z * fr[18] + fr[19] + bound_r) >= 0.0
                && (x * fr[20] + y * fr[21] + z * fr[22] + fr[23] + bound_r) >= 0.0
            );
            if visible { vis += 1; }

            let o = i * 8;
            out[o] = x;
            out[o + 1] = y;
            out[o + 2] = z;
            out[o + 3] = if visible { s.cr_scale[i] } else { 0.0 };
            out[o + 4] = ry;
            out[o + 5] = rx;
            out[o + 6] = s.cr_color[i];
            out[o + 7] = pulse;
            i += 1;
        }

        vis
    }
}
