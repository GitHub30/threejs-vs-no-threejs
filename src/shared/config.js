// Shared deterministic scene constants, lighting, palette, and benchmark scenarios

export const ARENA_FLOOR_Y = -44.0;
export const ARENA_FLOOR_RADIUS = 135.0;

// Normalized directional sunlight vector (warm alpine/coastal sun)
export const SUN_DIR = (() => {
  const x = 0.56, y = 0.72, z = 0.41;
  const len = Math.hypot(x, y, z);
  return [x / len, y / len, z / len];
})();

// Linear RGB lighting & atmospheric palette (shared identically between WGSL, GLSL, and Three.js)
export const COLORS = {
  skyZenith: [0.055, 0.215, 0.515],
  skyHorizon: [0.895, 0.805, 0.675],
  skyGround: [0.34, 0.44, 0.53],
  fog: [0.84, 0.785, 0.70],
  sunColor: [1.0, 0.94, 0.82],
  hemiSky: [0.42, 0.64, 0.92],
  hemiGround: [0.46, 0.39, 0.32],
  stoneLight: [0.82, 0.79, 0.74],
  stoneDark: [0.58, 0.56, 0.54],
  goldAccent: [0.95, 0.66, 0.22],
};

// 4 curated avian plumage colors (linear RGB)
export const BIRD_PALETTE = [
  [0.82, 0.16, 0.08], // Sunlit Vermilion / Coral
  [0.86, 0.46, 0.06], // Warm Ochre / Amber
  [0.04, 0.28, 0.38], // Deep Coastal Teal
  [0.90, 0.86, 0.78], // Alabaster Ivory
];

// 6 outer architectural pillars matching wasm/src/lib.rs exactly
export const PILLAR_RING_RADIUS = 38.0;
export const PILLARS = [
  { x: 38.0,  z: 0.0,         radius: 2.7, topY: 18.0 },
  { x: 19.0,  z: 32.908965,   radius: 2.5, topY: 13.0 },
  { x: -19.0, z: 32.908965,   radius: 2.8, topY: 22.0 },
  { x: -38.0, z: 0.0,         radius: 2.5, topY: 15.0 },
  { x: -19.0, z: -32.908965,  radius: 2.7, topY: 20.0 },
  { x: 19.0,  z: -32.908965,  radius: 2.5, topY: 12.0 },
];

export const OBELISK = {
  baseRadius: 4.6,
  topRadius: 1.9,
  topY: 20.0,
};

export const BOID_PARAMS = {
  separationWeight: 2.35,
  alignmentWeight: 1.45,
  cohesionWeight: 1.05,
  neighborRadius: 4.5,
  separationRadius: 2.0,
  minSpeed: 9.0,
  maxSpeed: 22.0,
  maxCellChecks: 16,
  gridMinX: -72.0,
  gridMinY: -36.0,
  gridMinZ: -72.0,
  cellSize: 4.5,
  gx: 32,
  gy: 16,
  gz: 32,
  pillarAvoidR: 5.6,
  obeliskAvoidR: 8.2,
};

// Peer budget per frame for GPU Compute mode (exact all-pairs when count <= 1024, rotating tiled block when > 1024)
export const COMPUTE_PEER_BUDGET = 1024;

export const SCENARIOS = {
  hybrid: {
    id: 'hybrid',
    label: 'Hybrid CPU+GPU',
    shortLabel: '1. Hybrid (Wasm SIMD)',
    title: '空間ハッシュ3Dボイド物理 + 動的インスタンシング',
    threeExampleRef: 'webgl_instancing_dynamic + webgpu_compute_birds',
    description:
      '毎フレーム3D空間ハッシュグリッドで近傍探索・障害物回避・フラスタムカリングを実行。自作エンジンはRust Wasm SIMD128＋32B圧縮バッファ＋WebGPUで実行し、Three.jsはJS空間ハッシュ＋InstancedMeshで同一物理を実行。',
    counts: [2048, 8192, 16384, 32768],
    defaultCount: 8192,
    benchCount: 16384,
    entityUnit: '羽',
  },
  'gpu-compute': {
    id: 'gpu-compute',
    label: 'GPU Compute',
    shortLabel: '2. GPU Compute (WGSL vs TSL)',
    title: '純GPUコンピュートシェーダー群衆シミュレーション',
    threeExampleRef: 'webgpu_compute_birds',
    description:
      'CPU-GPU間データ転送ゼロで全個体をGPU上で更新。自作エンジンはworkgroup共有メモリタイリング(var<workgroup>)＋1パス融合WGSL＋RenderBundle、Three.jsは公式WebGPURenderer＋TSL Computeを実行。',
    counts: [2048, 4096, 8192, 16384, 32768],
    defaultCount: 4096,
    benchCount: 8192,
    entityUnit: '羽',
  },
  'scene-graph': {
    id: 'scene-graph',
    label: 'Scene Graph',
    shortLabel: '3. Draw Call (RenderBundle)',
    title: '大量個別メッシュ描画・シーングラフ負荷テスト',
    threeExampleRef: 'webgpu_performance_renderbundle + webgl_instancing_performance',
    description:
      '数千の独立3Dクリスタルオブジェクトを毎フレーム個別ドローコールで描画。自作エンジンはRust Wasm SIMD4一括変換＋事前記録GPURenderBundle再生、Three.jsは標準THREE.Scene＋個別THREE.Mesh群で比較。',
    counts: [1000, 2500, 5000, 8000],
    defaultCount: 2500,
    benchCount: 4000,
    entityUnit: '体',
  },
};

export const ENGINE_FOOTPRINT = {
  custom: {
    label: '自作エンジン (Wasm SIMD + WebGPU)',
    rawKB: 59,
    gzipKB: 17,
    wasmKB: 10.8,
    deps: 0,
  },
  threeWebGPU: {
    label: 'Three.js v0.187 (WebGPU + TSL)',
    rawKB: 3839,
    gzipKB: 759,
    wasmKB: 0,
    deps: 1,
  },
  threeWebGL: {
    label: 'Three.js v0.187 (WebGL2)',
    rawKB: 667,
    gzipKB: 133,
    wasmKB: 0,
    deps: 1,
  },
};
