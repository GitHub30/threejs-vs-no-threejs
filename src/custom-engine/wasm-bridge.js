// Zero-dependency WebAssembly SIMD128 loader & linear memory bridge for Rust #![no_std] engine core

// Minimal valid Wasm module containing a v128.const + i32x4.splat instruction to feature-detect SIMD128
const WASM_SIMD_PROBE = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0,
  1, 5, 1, 96, 0, 1, 123,
  3, 2, 1, 0,
  10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11,
]);

export function supportsWasmSimd() {
  try {
    return typeof WebAssembly === 'object' && WebAssembly.validate(WASM_SIMD_PROBE);
  } catch {
    return false;
  }
}

let cachedPromise = null;

export function loadWasmCore() {
  if (!cachedPromise) {
    cachedPromise = instantiateWasmCore();
  }
  return cachedPromise;
}

async function instantiateWasmCore() {
  const useSimd = supportsWasmSimd();
  const url = new URL(
    useSimd ? '../../wasm/engine_simd.wasm' : '../../wasm/engine_scalar.wasm',
    import.meta.url,
  );

  const t0 = performance.now();
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch Wasm module (${response.status}): ${url}`);
  }
  const bytes = await response.arrayBuffer();
  const { instance } = await WebAssembly.instantiate(bytes);
  const loadMs = performance.now() - t0;

  return new WasmCoreBridge(instance.exports, bytes.byteLength, loadMs);
}

const MAX_ENTITIES = 65536;

class WasmCoreBridge {
  constructor(exports, byteLength, loadMs) {
    this.ex = exports;
    this.byteLength = byteLength;
    this.loadMs = loadMs;
    this.simdEnabled = exports.is_simd_enabled() === 1;
    this.label = this.simdEnabled
      ? `Rust Wasm SIMD128 (${(byteLength / 1024).toFixed(1)} KB)`
      : `Rust Wasm Scalar (${(byteLength / 1024).toFixed(1)} KB)`;

    const buf = exports.memory.buffer;

    // Zero-copy typed array views directly over static aligned Wasm linear memory
    this.px = new Float32Array(buf, exports.get_px_ptr(), MAX_ENTITIES);
    this.py = new Float32Array(buf, exports.get_py_ptr(), MAX_ENTITIES);
    this.pz = new Float32Array(buf, exports.get_pz_ptr(), MAX_ENTITIES);
    this.vx = new Float32Array(buf, exports.get_vx_ptr(), MAX_ENTITIES);
    this.vy = new Float32Array(buf, exports.get_vy_ptr(), MAX_ENTITIES);
    this.vz = new Float32Array(buf, exports.get_vz_ptr(), MAX_ENTITIES);
    this.phase = new Float32Array(buf, exports.get_phase_ptr(), MAX_ENTITIES);
    this.colorSeed = new Float32Array(buf, exports.get_color_ptr(), MAX_ENTITIES);

    this.gpuOut = new Float32Array(buf, exports.get_gpu_out_ptr(), MAX_ENTITIES * 8);
    this.frustum = new Float32Array(buf, exports.get_frustum_ptr(), 24);

    this.crRadius = new Float32Array(buf, exports.get_cr_radius_ptr(), MAX_ENTITIES);
    this.crAngle = new Float32Array(buf, exports.get_cr_angle_ptr(), MAX_ENTITIES);
    this.crBaseY = new Float32Array(buf, exports.get_cr_base_y_ptr(), MAX_ENTITIES);
    this.crOrbitSpd = new Float32Array(buf, exports.get_cr_orbit_spd_ptr(), MAX_ENTITIES);
    this.crSpinSpd = new Float32Array(buf, exports.get_cr_spin_spd_ptr(), MAX_ENTITIES);
    this.crScale = new Float32Array(buf, exports.get_cr_scale_ptr(), MAX_ENTITIES);
    this.crPhase = new Float32Array(buf, exports.get_cr_phase_ptr(), MAX_ENTITIES);
    this.crColor = new Float32Array(buf, exports.get_cr_color_ptr(), MAX_ENTITIES);
  }

  loadBoids(init) {
    this.px.set(init.px);
    this.py.set(init.py);
    this.pz.set(init.pz);
    this.vx.set(init.vx);
    this.vy.set(init.vy);
    this.vz.set(init.vz);
    this.phase.set(init.phase);
    this.colorSeed.set(init.colorSeed);
  }

  stepBoids(count, dt, ptrWorld, ptrMode, sepW, aliW, cohW, frustumPlanes, doCull = true) {
    if (doCull && frustumPlanes) {
      this.frustum.set(frustumPlanes);
    }
    return this.ex.step_boids(
      count,
      dt,
      ptrWorld[0],
      ptrWorld[1],
      ptrWorld[2],
      ptrMode,
      sepW,
      aliW,
      cohW,
      doCull ? 1 : 0,
    );
  }

  loadCrystals(init) {
    this.crRadius.set(init.radius);
    this.crAngle.set(init.angle);
    this.crBaseY.set(init.baseY);
    this.crOrbitSpd.set(init.orbitSpd);
    this.crSpinSpd.set(init.spinSpd);
    this.crScale.set(init.scale);
    this.crPhase.set(init.phase);
    this.crColor.set(init.color);
  }

  stepCrystals(count, time, frustumPlanes, doCull = true) {
    if (doCull && frustumPlanes) {
      this.frustum.set(frustumPlanes);
    }
    return this.ex.step_crystals(count, time, doCull ? 1 : 0);
  }
}
