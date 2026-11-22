// Zero-dependency WebGPU renderer for Osaka City.
// Implements:
//  - GPU Compute frustum culling + drawIndexedIndirect / drawIndirect for InstancedMesh
//  - Single-write batched DrawData & Skinned Bone Storage Buffers (zero per-draw uniform calls)
//  - Hardware texture_depth_2d_array CSM with world-origin texel snapping
//  - Native 4x MSAA with hardware resolveTarget (zero blitFramebuffer copy pass)
//  - Zero-resolve multisampled Compute SSAO + Compute Bloom + AP1 ACES Composite
//  - Gamma-aware GPU mipmap chain generation + anisotropic road/terrain filtering

import { BufferAttribute, BufferGeometry } from './geometry';
import {
  EngineRenderer,
  GPUAdapterLike,
  GPUBindGroupLayoutLike,
  GPUBindGroupLike,
  GPUBufferLike,
  GPUCanvasContextLike,
  GPUCommandEncoderLike,
  GPUComputePipelineLike,
  GPUDeviceLike,
  GPUNavigatorLike,
  GPUPipelineLayoutLike,
  GPURenderPassEncoderLike,
  GPURenderPipelineLike,
  GPUSamplerLike,
  GPUShaderModuleLike,
  GPUTextureLike,
  GPUTextureViewLike,
  GPU_BUFFER_USAGE,
  GPU_SHADER_STAGE,
  GPU_TEXTURE_USAGE,
  PostConfig,
  RendererOptions,
  ShadowConfig,
} from './gpuTypes';
import { Material, ShaderMaterial, Texture } from './materials';
import {
  ACESFilmicToneMapping,
  AdditiveBlending,
  BackSide,
  DoubleSide,
  FrontSide,
  Matrix4,
  RepeatWrapping,
  SRGBColorSpace,
  Vector3,
  extractFrustumPlanes,
  invertMat4Array,
  multiplyMat4Arrays,
  sphereIntersectsFrustum,
} from './math';
import {
  Camera,
  DirectionalLight,
  HemisphereLight,
  InstancedMesh,
  LineSegments,
  Mesh,
  Object3D,
  PerspectiveCamera,
  PointLight,
  Points,
  Scene,
  SkinnedMesh,
  SpotLight,
} from './scene';
import {
  DrawMode,
  FLAG_ALPHA_TO_COV,
  FLAG_AO_MAP,
  FLAG_BUMP_AO_UV1,
  FLAG_BUMP_MAP,
  FLAG_CAR_SHELL,
  FLAG_CRM_MAP,
  FLAG_EMISSIVE_MAP,
  FLAG_FLAT_SHADING,
  FLAG_FOG,
  FLAG_PERSON_OUTFIT,
  FLAG_RECEIVE_SHADOW,
  FLAG_TONEMAP_INLINE,
  FLAG_UNLIT,
  FLAG_VERTEX_COLORS,
  MAX_BONES,
  MAX_CASCADES,
} from './shaders';
import {
  WGSL_BLOOM_BLUR_COMPUTE,
  WGSL_BLOOM_EXTRACT_COMPUTE,
  WGSL_COMPOSITE,
  WGSL_CULL_COMPUTE,
  WGSL_LINES,
  WGSL_MIPMAP,
  WGSL_POINTS,
  WGSL_SKY,
  buildWgslPbr,
  buildWgslShadow,
  buildWgslSsaoCompute,
} from './wgpuShaders';

const DRAW_STRIDE_FLOATS = 48; // 192 bytes per DrawData
const DRAW_STRIDE_BYTES = DRAW_STRIDE_FLOATS * 4;
const SCENE_UNIFORM_BYTES = 656; // 164 floats
const SKY_UNIFORM_BYTES = 224;   // 56 floats
const INITIAL_CULL_CAP = 256;
const MAX_POINTS_SLOTS = 16;
const MAX_LINES_SLOTS = 16;

interface GeoGpu {
  vbuf: GPUBufferLike;
  skinBuf: GPUBufferLike | null;
  ibuf: GPUBufferLike | null;
  indexFormat: 'uint16' | 'uint32';
  indexCount: number;
  vertexCount: number;
  layoutVersion: number;
  disposedVersion: number;
  attrVersions: Map<string, number>;
  indexVersion: number;
}

interface TexGpu {
  tex: GPUTextureLike;
  view: GPUTextureViewLike;
  version: number;
  disposedVersion: number;
  isSrgb: boolean;
  wrapRepeat: boolean;
  aniso: number;
}

interface MatBgCache {
  bg: GPUBindGroupLike;
  key: string;
}

interface InstGpu {
  total: number;
  geoId: number;
  geoDisp: number;
  geoLayout: number;
  matBuf: GPUBufferLike;
  colBuf: GPUBufferLike;
  sphereBuf: GPUBufferLike;
  visBuf: GPUBufferLike;
  indirectBuf: GPUBufferLike;
  indirectTemplate: GPUBufferLike;
  slotUniformBuf: GPUBufferLike;
  renderBg: GPUBindGroupLike;
  cullBg: GPUBindGroupLike;
  matVer: number;
  colVer: number;
  worldMat: Float32Array;
  worldMatValid: boolean;
  isIndexed: boolean;
  drawCount: number;
  spheresCpu: Float32Array;
  colsCpu: Float32Array;
  slotCpu: Uint32Array;
}

interface DynPointsGpu {
  buf: GPUBufferLike;
  cap: number;
  posVer: number;
  colVer: number;
  sizeVer: number;
  alphaVer: number;
  count: number;
  scratch: Float32Array;
}

interface DynLinesGpu {
  buf: GPUBufferLike;
  cap: number;
  posVer: number;
  count: number;
}

interface DrawItem {
  obj: Mesh;
  geo: BufferGeometry;
  mat: Material;
  start: number;
  count: number;
  mode: DrawMode;
  modeId: number;
  sortKey: number;
  depth: number;
  drawIdx: number;
  sx: number;
  sy: number;
  sz: number;
  sr: number;
}

interface CullJob {
  inst: InstGpu;
  slot: number;
  planes: Float32Array;
}

// Module-level zero-allocation scratch buffers
const _vpGl = new Float32Array(16);
const _vpZo = new Float32Array(16);
const _camPlanes = new Float32Array(24);
const _shPlanesAll = [
  new Float32Array(24),
  new Float32Array(24),
  new Float32Array(24),
  new Float32Array(24),
];
const _lightView = new Float32Array(16);
const _lightProj = new Float32Array(16);
const _lightVpGl = new Float32Array(16);
const _meshWorldInv = new Float32Array(16);
const _tmpMat4 = new Matrix4();
const _vSun = new Vector3();
const _vUp = new Vector3(0, 1, 0);
const _vAltUp = new Vector3(1, 0, 0);
const _vRight = new Vector3();
const _vLightUp = new Vector3();
const _vSpotDir = new Vector3();
const _cx = new Float32Array(8);
const _cy = new Float32Array(8);
const _cz = new Float32Array(8);
const _wx = new Float32Array(8);
const _wy = new Float32Array(8);
const _wz = new Float32Array(8);

const sortOpaque = (a: DrawItem, b: DrawItem) => a.sortKey - b.sortKey;
const sortTransparent = (a: DrawItem, b: DrawItem) => (a.obj.renderOrder - b.obj.renderOrder) || (b.depth - a.depth);

function isObjVisible(o: Object3D): boolean {
  let cur: Object3D | null = o;
  while (cur) {
    if (!cur.visible) return false;
    cur = cur.parent;
  }
  return true;
}

/** Convert OpenGL [-1, 1] clip-Z projection matrix into WebGPU [0, 1] clip-Z. */
function toZeroOneClipZ(out: Float32Array, m: Float32Array, outOffset = 0): void {
  for (let i = 0; i < 16; i++) out[outOffset + i] = m[i];
  out[outOffset + 2]  = 0.5 * (m[2]  + m[3]);
  out[outOffset + 6]  = 0.5 * (m[6]  + m[7]);
  out[outOffset + 10] = 0.5 * (m[10] + m[11]);
  out[outOffset + 14] = 0.5 * (m[14] + m[15]);
}

function align4(n: number): number {
  return (n + 3) & ~3;
}

function paddedView4(src: ArrayBufferView): Uint8Array {
  const bytes = new Uint8Array(src.buffer, src.byteOffset, src.byteLength);
  if ((bytes.byteLength & 3) === 0) return bytes;
  const out = new Uint8Array(align4(bytes.byteLength));
  out.set(bytes);
  return out;
}

/** Return [scale, isSigned] for WebGL/Three-style normalized integer BufferAttributes. */
function attrNormScale(attr: BufferAttribute | undefined): { scale: number; signed: boolean } {
  if (!attr || !attr.normalized) return { scale: 1, signed: false };
  const a = attr.array;
  if (a instanceof Int8Array) return { scale: 1 / 127, signed: true };
  if (a instanceof Uint8Array || a instanceof Uint8ClampedArray) return { scale: 1 / 255, signed: false };
  if (a instanceof Int16Array) return { scale: 1 / 32767, signed: true };
  if (a instanceof Uint16Array) return { scale: 1 / 65535, signed: false };
  if (a instanceof Int32Array) return { scale: 1 / 2147483647, signed: true };
  if (a instanceof Uint32Array) return { scale: 1 / 4294967295, signed: false };
  return { scale: 1, signed: false };
}

export class WebGPURenderer implements EngineRenderer {
  readonly backend = 'webgpu' as const;
  readonly domElement: HTMLCanvasElement;
  readonly capabilities: { maxTextureSize: number; maxAnisotropy: number; maxSamples: number };

  shadowMap = { enabled: true, type: 2 };
  toneMapping = ACESFilmicToneMapping;
  toneMappingExposure = 0.95;

  postConfig: PostConfig = { enabled: false, aoScale: 0, bloomStrength: 0.32, reflections: false };
  shadowConfig: ShadowConfig = { enabled: true, mapSize: 1024, cascades: null };

  info = {
    render: { calls: 0, triangles: 0 },
    memory: { geometries: 0, textures: 0 },
    programs: [] as unknown[],
  };

  private device: GPUDeviceLike;
  private ctx!: GPUCanvasContextLike;
  private canvasFormat: string;
  private hdrFormat = 'rgba16float';
  private uAlign: number;
  private antialias: boolean;

  private pixelRatio = 1;
  private cssW = 1;
  private cssH = 1;
  private fbW = 1;
  private fbH = 1;
  private msaaSamples = 4;
  private deviceLost = false;

  // Shader modules & pipeline layouts
  private pbrMods!: Record<DrawMode, GPUShaderModuleLike>;
  private shadowMods!: Record<DrawMode, GPUShaderModuleLike>;
  private skyMod!: GPUShaderModuleLike;
  private pointsMod!: GPUShaderModuleLike;
  private linesMod!: GPUShaderModuleLike;
  private compMod!: GPUShaderModuleLike;
  private mipMod!: GPUShaderModuleLike;

  private sceneBgl!: GPUBindGroupLayoutLike;
  private shadowPassBgl!: GPUBindGroupLayoutLike;
  private drawBoneBgl!: GPUBindGroupLayoutLike;
  private matBgl!: GPUBindGroupLayoutLike;
  private instBgl!: GPUBindGroupLayoutLike;
  private cullBgl!: GPUBindGroupLayoutLike;
  private skyBgl!: GPUBindGroupLayoutLike;
  private pointsBgl!: GPUBindGroupLayoutLike;
  private linesBgl!: GPUBindGroupLayoutLike;
  private ssaoMsaaBgl!: GPUBindGroupLayoutLike;
  private ssaoSingleBgl!: GPUBindGroupLayoutLike;
  private bloomBgl!: GPUBindGroupLayoutLike;
  private compBgl!: GPUBindGroupLayoutLike;
  private mipBgl!: GPUBindGroupLayoutLike;

  private pbrLayouts!: Record<DrawMode, GPUPipelineLayoutLike>;
  private shadowLayouts!: Record<DrawMode, GPUPipelineLayoutLike>;
  private skyLayout!: GPUPipelineLayoutLike;
  private pointsLayout!: GPUPipelineLayoutLike;
  private linesLayout!: GPUPipelineLayoutLike;
  private pbrPipeCache = new Map<number, GPURenderPipelineLike>();
  private shadowPipeCache = new Map<number, GPURenderPipelineLike>();
  private skyPipeCache = new Map<number, GPURenderPipelineLike>();
  private pointsPipeCache = new Map<number, GPURenderPipelineLike>();
  private linesPipeCache = new Map<number, GPURenderPipelineLike>();
  private compPipe!: GPURenderPipelineLike;
  private mipLinPipe!: GPURenderPipelineLike;
  private mipSrgbPipe!: GPURenderPipelineLike;

  private cullPipe!: GPUComputePipelineLike;
  private ssaoMsaaPipe!: GPUComputePipelineLike;
  private ssaoSinglePipe!: GPUComputePipelineLike;
  private bloomExtPipe!: GPUComputePipelineLike;
  private bloomBlurPipe!: GPUComputePipelineLike;

  // Shared samplers & 1x1 fallback textures
  private sampRepeatAniso!: GPUSamplerLike;
  private sampClampAniso!: GPUSamplerLike;
  private sampRepeatLin!: GPUSamplerLike;
  private sampClampLin!: GPUSamplerLike;
  private sampNearestClamp!: GPUSamplerLike;
  private sampShadowCmp!: GPUSamplerLike;

  private whiteTex!: GPUTextureLike;
  private whiteView!: GPUTextureViewLike;
  private blackTex!: GPUTextureLike;
  private blackView!: GPUTextureViewLike;
  private clearTex!: GPUTextureLike;
  private clearView!: GPUTextureViewLike;
  private dummyShadowTex!: GPUTextureLike;
  private dummyShadowView!: GPUTextureViewLike;

  // Per-frame uniform & storage buffers
  private sceneUniformBuf!: GPUBufferLike;
  private sceneUniformCpu = new Float32Array(SCENE_UNIFORM_BYTES / 4);
  private sceneBg!: GPUBindGroupLike;
  private sceneBgShadowView: GPUTextureViewLike | null = null;

  private drawStorageCap = 2048;
  private drawStorageBuf!: GPUBufferLike;
  private drawRawBuf = new ArrayBuffer(2048 * DRAW_STRIDE_BYTES);
  private drawF32 = new Float32Array(this.drawRawBuf);
  private drawU32 = new Uint32Array(this.drawRawBuf);

  private boneStorageCap = 2048; // matrices
  private boneStorageBuf!: GPUBufferLike;
  private boneCpu = new Float32Array(2048 * 16);
  private drawBoneBg!: GPUBindGroupLike;

  private shadowPassUniformBuf!: GPUBufferLike;
  private shadowPassBg!: GPUBindGroupLike;
  private cullParamsCap = INITIAL_CULL_CAP;
  private cullParamsBuf!: GPUBufferLike;
  private cullParamsCpu!: ArrayBuffer;
  private cullParamsF32!: Float32Array;
  private cullParamsU32!: Uint32Array;

  private skyUniformBuf!: GPUBufferLike;
  private skyUniformCpu = new Float32Array(SKY_UNIFORM_BYTES / 4);
  private skyBg!: GPUBindGroupLike;

  private pointsUniformBuf!: GPUBufferLike;
  private pointsUniformCpu!: Float32Array;
  private pointsBg!: GPUBindGroupLike;

  private linesUniformBuf!: GPUBufferLike;
  private linesUniformCpu!: Float32Array;
  private linesBg!: GPUBindGroupLike;

  private ssaoUniformBuf!: GPUBufferLike;
  private ssaoUniformRaw = new ArrayBuffer(48);
  private ssaoUniformF32 = new Float32Array(this.ssaoUniformRaw);
  private ssaoUniformU32 = new Uint32Array(this.ssaoUniformRaw);
  private bloomExtUniformBuf!: GPUBufferLike;
  private bloomExtUniformCpu = new Float32Array(4);
  private bloomBlurUniformBuf!: GPUBufferLike;
  private bloomBlurUniformCpu!: Float32Array;
  private compUniformBuf!: GPUBufferLike;
  private compUniformCpu = new Float32Array(8);

  // Shadow map array
  private shadowArrayTex: GPUTextureLike | null = null;
  private shadowArrayView: GPUTextureViewLike | null = null;
  private shadowLayerViews: GPUTextureViewLike[] = [];
  private shadowTexSize = 0;
  private shadowLayers = 0;
  private shadowMatsZo = new Float32Array(MAX_CASCADES * 16);
  private cascadeSplits = new Float32Array(MAX_CASCADES);
  private cascadeBiases = new Float32Array(MAX_CASCADES);
  private cascadeDepthBiases = new Float32Array(MAX_CASCADES);
  private shadowFocusX = 0;
  private shadowFocusZ = 0;
  private shadowLowRange2 = Infinity;

  // Render & Post-process targets
  private tgtW = 0;
  private tgtH = 0;
  private tgtHdr = false;
  private tgtMsaa = 1;
  private tgtAoW = 0;
  private tgtAoH = 0;

  private msaaColorTex: GPUTextureLike | null = null;
  private msaaColorView: GPUTextureViewLike | null = null;
  private depthTex: GPUTextureLike | null = null;
  private depthView: GPUTextureViewLike | null = null;
  private sceneColorTex: GPUTextureLike | null = null;
  private sceneColorView: GPUTextureViewLike | null = null;
  private aoTex: GPUTextureLike | null = null;
  private aoSampleView: GPUTextureViewLike | null = null;
  private aoStorageView: GPUTextureViewLike | null = null;
  private bloomTexA: GPUTextureLike | null = null;
  private bloomViewA: GPUTextureViewLike | null = null;
  private bloomTexB: GPUTextureLike | null = null;
  private bloomViewB: GPUTextureViewLike | null = null;

  private ssaoBg: GPUBindGroupLike | null = null;
  private bloomExtBg: GPUBindGroupLike | null = null;
  private bloomBlurHBg: GPUBindGroupLike | null = null;
  private bloomBlurVBg: GPUBindGroupLike | null = null;
  private compBg: GPUBindGroupLike | null = null;

  // GPU resource pools
  private geoCache = new Map<number, GeoGpu>();
  private texCache = new Map<number, TexGpu>();
  private matBgCache = new Map<number, MatBgCache>();
  private instCache = new Map<number, InstGpu>();
  private pointsCache = new Map<number, DynPointsGpu>();
  private linesCache = new Map<number, DynLinesGpu>();

  // Persistent flattened scene lists
  private cachedSceneId = -1;
  private cachedStructVer = -1;
  private flatMeshes: Mesh[] = [];
  private flatPoints: Points[] = [];
  private flatLines: LineSegments[] = [];
  private flatSky: Mesh | null = null;
  private sunLight: DirectionalLight | null = null;
  private hemiLight: HemisphereLight | null = null;
  private spotLight: SpotLight | null = null;
  private pointLight: PointLight | null = null;

  // Reusable per-frame queues
  private opaqueQueue: DrawItem[] = [];
  private transparentQueue: DrawItem[] = [];
  private shadowQueue: DrawItem[] = [];
  private drawPool: DrawItem[] = [];
  private poolIdx = 0;
  private cullJobs: CullJob[] = [];
  private activeInstsSet = new Set<InstGpu>();
  private skyTop = [0.18, 0.45, 0.8];
  private skyHor = [0.72, 0.85, 0.94];

  private scratchCanvas: HTMLCanvasElement | null = null;
  private pendingMips: { tex: GPUTextureLike; mipCount: number; isSrgb: boolean }[] = [];

  private constructor(
    opts: RendererOptions,
    device: GPUDeviceLike,
    canvasFormat: string,
  ) {
    this.domElement = opts.canvas;
    this.device = device;
    this.canvasFormat = canvasFormat;
    this.antialias = opts.antialias ?? true;
    this.msaaSamples = this.antialias ? 4 : 1;
    this.uAlign = Math.max(256, device.limits.minUniformBufferOffsetAlignment || 256);
    this.fbW = Math.max(1, opts.canvas.width || 1);
    this.fbH = Math.max(1, opts.canvas.height || 1);
    this.cssW = this.fbW;
    this.cssH = this.fbH;
    this.bloomBlurUniformCpu = new Float32Array((2 * this.uAlign) / 4);

    this.capabilities = {
      maxTextureSize: Math.min(8192, device.limits.maxTextureDimension2D || 4096),
      maxAnisotropy: 8,
      maxSamples: 4,
    };

    device.lost.then((info) => {
      this.deviceLost = true;
      console.warn('[WebGPURenderer] Device lost:', info?.reason, info?.message);
    });
  }

  /**
   * Async factory: acquires WebGPU adapter & device, compiles & validates core pipelines
   * BEFORE acquiring the canvas 'webgpu' context so failed initialization leaves the canvas
   * untouched for seamless WebGL2 fallback.
   */
  static async create(opts: RendererOptions, allowFallback = false): Promise<WebGPURenderer> {
    const gpu = (navigator as unknown as { gpu?: GPUNavigatorLike }).gpu;
    if (!gpu) throw new Error('WebGPU not supported');

    const pref = opts.powerPreference === 'low-power' ? 'low-power' : 'high-performance';
    let adapter: GPUAdapterLike | null = await gpu.requestAdapter({ powerPreference: pref });
    if (!adapter && allowFallback) {
      adapter = await gpu.requestAdapter({ forceFallbackAdapter: true });
    }
    if (!adapter) throw new Error('No WebGPU adapter available');
    if (!allowFallback && adapter.info?.isFallbackAdapter) {
      throw new Error('Only CPU fallback WebGPU adapter available; preferring hardware WebGL2');
    }

    const device = await adapter.requestDevice();
    const format = gpu.getPreferredCanvasFormat();

    const renderer = new WebGPURenderer(opts, device, format);

    // Validate all shader compilation & core pipeline creation before binding the canvas
    device.pushErrorScope('validation');
    await renderer.initResources();
    const err = await device.popErrorScope();
    if (err) {
      device.destroy();
      throw new Error(`WebGPU validation failed during init: ${err.message}`);
    }

    const ctx = opts.canvas.getContext('webgpu') as unknown as GPUCanvasContextLike | null;
    if (!ctx) {
      device.destroy();
      throw new Error('Failed to acquire WebGPU canvas context');
    }
    renderer.ctx = ctx;
    renderer.configureCanvas();
    return renderer;
  }

  getContext(): GPUCanvasContextLike {
    return this.ctx;
  }

  setPixelRatio(pr: number): void {
    this.pixelRatio = Math.max(0.5, pr);
  }

  setSize(w: number, h: number, updateStyle = true): void {
    this.cssW = Math.max(1, w);
    this.cssH = Math.max(1, h);
    const nw = Math.max(1, Math.round(this.cssW * this.pixelRatio));
    const nh = Math.max(1, Math.round(this.cssH * this.pixelRatio));
    if (this.fbW !== nw || this.fbH !== nh) {
      this.fbW = nw;
      this.fbH = nh;
      this.domElement.width = nw;
      this.domElement.height = nh;
      if (this.ctx) this.configureCanvas();
    }
    if (updateStyle) {
      this.domElement.style.width = `${this.cssW}px`;
      this.domElement.style.height = `${this.cssH}px`;
    }
  }

  private configureCanvas(): void {
    this.ctx.configure({
      device: this.device,
      format: this.canvasFormat,
      alphaMode: 'opaque',
      usage: GPU_TEXTURE_USAGE.RENDER_ATTACHMENT | GPU_TEXTURE_USAGE.COPY_SRC,
    });
  }

  // =========================================================================
  // Core Resource, Layout & Pipeline Initialization
  // =========================================================================

  private async initResources(): Promise<void> {
    const dev = this.device;

    // Samplers
    this.sampRepeatAniso = dev.createSampler({
      addressModeU: 'repeat', addressModeV: 'repeat',
      magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', maxAnisotropy: 8,
    });
    this.sampClampAniso = dev.createSampler({
      addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge',
      magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', maxAnisotropy: 8,
    });
    this.sampRepeatLin = dev.createSampler({
      addressModeU: 'repeat', addressModeV: 'repeat',
      magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear',
    });
    this.sampClampLin = dev.createSampler({
      addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge',
      magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear',
    });
    this.sampNearestClamp = dev.createSampler({
      addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge',
      magFilter: 'nearest', minFilter: 'nearest',
    });
    this.sampShadowCmp = dev.createSampler({
      addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge',
      magFilter: 'linear', minFilter: 'linear', compare: 'less-equal',
    });

    // 1x1 fallback textures
    const make1x1 = (rgba: [number, number, number, number]) => {
      const tex = dev.createTexture({
        size: [1, 1],
        format: 'rgba8unorm',
        usage: GPU_TEXTURE_USAGE.TEXTURE_BINDING | GPU_TEXTURE_USAGE.COPY_DST,
      });
      dev.queue.writeTexture({ texture: tex }, new Uint8Array(rgba), { bytesPerRow: 4 }, [1, 1]);
      return { tex, view: tex.createView() };
    };
    const w = make1x1([255, 255, 255, 255]); this.whiteTex = w.tex; this.whiteView = w.view;
    const b = make1x1([0, 0, 0, 255]); this.blackTex = b.tex; this.blackView = b.view;
    const c = make1x1([0, 0, 0, 0]); this.clearTex = c.tex; this.clearView = c.view;

    this.dummyShadowTex = dev.createTexture({
      size: [1, 1, 1],
      format: 'depth24plus',
      usage: GPU_TEXTURE_USAGE.TEXTURE_BINDING | GPU_TEXTURE_USAGE.RENDER_ATTACHMENT,
    });
    this.dummyShadowView = this.dummyShadowTex.createView({ dimension: '2d-array', baseArrayLayer: 0, arrayLayerCount: 1 });

    // Bind Group Layouts
    this.sceneBgl = dev.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPU_SHADER_STAGE.VERTEX | GPU_SHADER_STAGE.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPU_SHADER_STAGE.FRAGMENT, texture: { sampleType: 'depth', viewDimension: '2d-array' } },
        { binding: 2, visibility: GPU_SHADER_STAGE.FRAGMENT, sampler: { type: 'comparison' } },
      ],
    });
    this.shadowPassBgl = dev.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPU_SHADER_STAGE.VERTEX, buffer: { type: 'uniform', hasDynamicOffset: true } },
      ],
    });
    this.drawBoneBgl = dev.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPU_SHADER_STAGE.VERTEX | GPU_SHADER_STAGE.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPU_SHADER_STAGE.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });
    this.matBgl = dev.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPU_SHADER_STAGE.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPU_SHADER_STAGE.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPU_SHADER_STAGE.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 3, visibility: GPU_SHADER_STAGE.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 4, visibility: GPU_SHADER_STAGE.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 5, visibility: GPU_SHADER_STAGE.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 6, visibility: GPU_SHADER_STAGE.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 7, visibility: GPU_SHADER_STAGE.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 8, visibility: GPU_SHADER_STAGE.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    });
    this.instBgl = dev.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPU_SHADER_STAGE.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPU_SHADER_STAGE.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPU_SHADER_STAGE.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPU_SHADER_STAGE.VERTEX, buffer: { type: 'uniform', hasDynamicOffset: true } },
      ],
    });
    this.cullBgl = dev.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPU_SHADER_STAGE.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true } },
        { binding: 1, visibility: GPU_SHADER_STAGE.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPU_SHADER_STAGE.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPU_SHADER_STAGE.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    this.skyBgl = dev.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPU_SHADER_STAGE.VERTEX | GPU_SHADER_STAGE.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });
    this.pointsBgl = dev.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPU_SHADER_STAGE.VERTEX | GPU_SHADER_STAGE.FRAGMENT, buffer: { type: 'uniform', hasDynamicOffset: true } },
      ],
    });
    this.linesBgl = dev.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPU_SHADER_STAGE.VERTEX | GPU_SHADER_STAGE.FRAGMENT, buffer: { type: 'uniform', hasDynamicOffset: true } },
      ],
    });
    this.ssaoMsaaBgl = dev.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPU_SHADER_STAGE.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPU_SHADER_STAGE.COMPUTE, texture: { sampleType: 'depth', multisampled: true } },
        { binding: 2, visibility: GPU_SHADER_STAGE.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba8unorm' } },
      ],
    });
    this.ssaoSingleBgl = dev.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPU_SHADER_STAGE.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPU_SHADER_STAGE.COMPUTE, texture: { sampleType: 'depth', multisampled: false } },
        { binding: 2, visibility: GPU_SHADER_STAGE.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba8unorm' } },
      ],
    });
    this.bloomBgl = dev.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPU_SHADER_STAGE.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true } },
        { binding: 1, visibility: GPU_SHADER_STAGE.COMPUTE, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPU_SHADER_STAGE.COMPUTE, sampler: { type: 'filtering' } },
        { binding: 3, visibility: GPU_SHADER_STAGE.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
      ],
    });
    this.compBgl = dev.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPU_SHADER_STAGE.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPU_SHADER_STAGE.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPU_SHADER_STAGE.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 3, visibility: GPU_SHADER_STAGE.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 4, visibility: GPU_SHADER_STAGE.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    });
    this.mipBgl = dev.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPU_SHADER_STAGE.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPU_SHADER_STAGE.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    });

    // Pipeline Layouts
    this.pbrLayouts = {
      static: dev.createPipelineLayout({ bindGroupLayouts: [this.sceneBgl, this.drawBoneBgl, this.matBgl] }),
      instanced: dev.createPipelineLayout({ bindGroupLayouts: [this.sceneBgl, this.drawBoneBgl, this.matBgl, this.instBgl] }),
      skinned: dev.createPipelineLayout({ bindGroupLayouts: [this.sceneBgl, this.drawBoneBgl, this.matBgl] }),
    };
    this.shadowLayouts = {
      static: dev.createPipelineLayout({ bindGroupLayouts: [this.shadowPassBgl, this.drawBoneBgl, this.matBgl] }),
      instanced: dev.createPipelineLayout({ bindGroupLayouts: [this.shadowPassBgl, this.drawBoneBgl, this.matBgl, this.instBgl] }),
      skinned: dev.createPipelineLayout({ bindGroupLayouts: [this.shadowPassBgl, this.drawBoneBgl, this.matBgl] }),
    };
    this.skyLayout = dev.createPipelineLayout({ bindGroupLayouts: [this.skyBgl] });
    this.pointsLayout = dev.createPipelineLayout({ bindGroupLayouts: [this.pointsBgl] });
    this.linesLayout = dev.createPipelineLayout({ bindGroupLayouts: [this.linesBgl] });

    // Compile all WGSL shader modules
    this.pbrMods = {
      static: dev.createShaderModule({ code: buildWgslPbr('static') }),
      instanced: dev.createShaderModule({ code: buildWgslPbr('instanced') }),
      skinned: dev.createShaderModule({ code: buildWgslPbr('skinned') }),
    };
    this.shadowMods = {
      static: dev.createShaderModule({ code: buildWgslShadow('static') }),
      instanced: dev.createShaderModule({ code: buildWgslShadow('instanced') }),
      skinned: dev.createShaderModule({ code: buildWgslShadow('skinned') }),
    };
    this.skyMod = dev.createShaderModule({ code: WGSL_SKY });
    this.pointsMod = dev.createShaderModule({ code: WGSL_POINTS });
    this.linesMod = dev.createShaderModule({ code: WGSL_LINES });
    this.compMod = dev.createShaderModule({ code: WGSL_COMPOSITE });
    this.mipMod = dev.createShaderModule({ code: WGSL_MIPMAP });

    const cullMod = dev.createShaderModule({ code: WGSL_CULL_COMPUTE });
    const ssaoMsaaMod = dev.createShaderModule({ code: buildWgslSsaoCompute(true) });
    const ssaoSingleMod = dev.createShaderModule({ code: buildWgslSsaoCompute(false) });
    const bloomExtMod = dev.createShaderModule({ code: WGSL_BLOOM_EXTRACT_COMPUTE });
    const bloomBlurMod = dev.createShaderModule({ code: WGSL_BLOOM_BLUR_COMPUTE });

    // Pre-compile core compute & post pipelines asynchronously in parallel!
    const [cullPipe, ssaoMsaaPipe, ssaoSinglePipe, bloomExtPipe, bloomBlurPipe, compPipe, mipLinPipe, mipSrgbPipe] = await Promise.all([
      dev.createComputePipelineAsync({
        layout: dev.createPipelineLayout({ bindGroupLayouts: [this.cullBgl] }),
        compute: { module: cullMod, entryPoint: 'csMain' },
      }),
      dev.createComputePipelineAsync({
        layout: dev.createPipelineLayout({ bindGroupLayouts: [this.ssaoMsaaBgl] }),
        compute: { module: ssaoMsaaMod, entryPoint: 'csMain' },
      }),
      dev.createComputePipelineAsync({
        layout: dev.createPipelineLayout({ bindGroupLayouts: [this.ssaoSingleBgl] }),
        compute: { module: ssaoSingleMod, entryPoint: 'csMain' },
      }),
      dev.createComputePipelineAsync({
        layout: dev.createPipelineLayout({ bindGroupLayouts: [this.bloomBgl] }),
        compute: { module: bloomExtMod, entryPoint: 'csMain' },
      }),
      dev.createComputePipelineAsync({
        layout: dev.createPipelineLayout({ bindGroupLayouts: [this.bloomBgl] }),
        compute: { module: bloomBlurMod, entryPoint: 'csMain' },
      }),
      dev.createRenderPipelineAsync({
        layout: dev.createPipelineLayout({ bindGroupLayouts: [this.compBgl] }),
        vertex: { module: this.compMod, entryPoint: 'vsMain' },
        fragment: { module: this.compMod, entryPoint: 'fsMain', targets: [{ format: this.canvasFormat }] },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
      }),
      dev.createRenderPipelineAsync({
        layout: dev.createPipelineLayout({ bindGroupLayouts: [this.mipBgl] }),
        vertex: { module: this.mipMod, entryPoint: 'vsMain' },
        fragment: { module: this.mipMod, entryPoint: 'fsLinear', targets: [{ format: 'rgba8unorm' }] },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
      }),
      dev.createRenderPipelineAsync({
        layout: dev.createPipelineLayout({ bindGroupLayouts: [this.mipBgl] }),
        vertex: { module: this.mipMod, entryPoint: 'vsMain' },
        fragment: { module: this.mipMod, entryPoint: 'fsSrgb', targets: [{ format: 'rgba8unorm' }] },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
      }),
    ]);
    this.cullPipe = cullPipe;
    this.ssaoMsaaPipe = ssaoMsaaPipe;
    this.ssaoSinglePipe = ssaoSinglePipe;
    this.bloomExtPipe = bloomExtPipe;
    this.bloomBlurPipe = bloomBlurPipe;
    this.compPipe = compPipe;
    this.mipLinPipe = mipLinPipe;
    this.mipSrgbPipe = mipSrgbPipe;
    this.info.programs.push(cullPipe, ssaoMsaaPipe, ssaoSinglePipe, bloomExtPipe, bloomBlurPipe, compPipe, mipLinPipe, mipSrgbPipe);

    // Allocate persistent uniform & storage buffers
    this.sceneUniformBuf = dev.createBuffer({ size: SCENE_UNIFORM_BYTES, usage: GPU_BUFFER_USAGE.UNIFORM | GPU_BUFFER_USAGE.COPY_DST });
    this.drawStorageBuf = dev.createBuffer({ size: this.drawStorageCap * DRAW_STRIDE_BYTES, usage: GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_DST });
    this.boneStorageBuf = dev.createBuffer({ size: this.boneStorageCap * 64, usage: GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_DST });
    this.rebuildDrawBoneBg();
    this.rebuildSceneBg(this.dummyShadowView);

    this.shadowPassUniformBuf = dev.createBuffer({ size: MAX_CASCADES * this.uAlign, usage: GPU_BUFFER_USAGE.UNIFORM | GPU_BUFFER_USAGE.COPY_DST });
    this.shadowPassBg = dev.createBindGroup({
      layout: this.shadowPassBgl,
      entries: [{ binding: 0, resource: { buffer: this.shadowPassUniformBuf, offset: 0, size: 64 } }],
    });

    this.cullParamsBuf = dev.createBuffer({ size: this.cullParamsCap * this.uAlign, usage: GPU_BUFFER_USAGE.UNIFORM | GPU_BUFFER_USAGE.COPY_DST });
    this.cullParamsCpu = new ArrayBuffer(this.cullParamsCap * this.uAlign);
    this.cullParamsF32 = new Float32Array(this.cullParamsCpu);
    this.cullParamsU32 = new Uint32Array(this.cullParamsCpu);

    this.skyUniformBuf = dev.createBuffer({ size: SKY_UNIFORM_BYTES, usage: GPU_BUFFER_USAGE.UNIFORM | GPU_BUFFER_USAGE.COPY_DST });
    this.skyBg = dev.createBindGroup({ layout: this.skyBgl, entries: [{ binding: 0, resource: { buffer: this.skyUniformBuf } }] });

    this.pointsUniformBuf = dev.createBuffer({ size: MAX_POINTS_SLOTS * this.uAlign, usage: GPU_BUFFER_USAGE.UNIFORM | GPU_BUFFER_USAGE.COPY_DST });
    this.pointsUniformCpu = new Float32Array((MAX_POINTS_SLOTS * this.uAlign) / 4);
    this.pointsBg = dev.createBindGroup({
      layout: this.pointsBgl,
      entries: [{ binding: 0, resource: { buffer: this.pointsUniformBuf, offset: 0, size: 224 } }],
    });

    this.linesUniformBuf = dev.createBuffer({ size: MAX_LINES_SLOTS * this.uAlign, usage: GPU_BUFFER_USAGE.UNIFORM | GPU_BUFFER_USAGE.COPY_DST });
    this.linesUniformCpu = new Float32Array((MAX_LINES_SLOTS * this.uAlign) / 4);
    this.linesBg = dev.createBindGroup({
      layout: this.linesBgl,
      entries: [{ binding: 0, resource: { buffer: this.linesUniformBuf, offset: 0, size: 160 } }],
    });

    this.ssaoUniformBuf = dev.createBuffer({ size: 64, usage: GPU_BUFFER_USAGE.UNIFORM | GPU_BUFFER_USAGE.COPY_DST });
    this.bloomExtUniformBuf = dev.createBuffer({ size: this.uAlign, usage: GPU_BUFFER_USAGE.UNIFORM | GPU_BUFFER_USAGE.COPY_DST });
    this.bloomBlurUniformBuf = dev.createBuffer({ size: 2 * this.uAlign, usage: GPU_BUFFER_USAGE.UNIFORM | GPU_BUFFER_USAGE.COPY_DST });
    this.compUniformBuf = dev.createBuffer({ size: 32, usage: GPU_BUFFER_USAGE.UNIFORM | GPU_BUFFER_USAGE.COPY_DST });

    // Pre-warm core PBR + Shadow + Sky + Points + Lines pipeline variants asynchronously in parallel (zero frame-1 stutter!)
    let savedLevel = 'high';
    try { savedLevel = localStorage.getItem('velabay.graphics') || 'high'; } catch { /* ignore */ }
    const defaultHdr = savedLevel !== 'low';
    const s = this.msaaSamples;
    const warmTasks: Promise<void>[] = [
      this.getPbrPipelineAsync('static', 0, FrontSide, 0, true, 0, false, s, defaultHdr).then(() => {}),
      this.getPbrPipelineAsync('static', 0, FrontSide, 0, true, 1, false, s, defaultHdr).then(() => {}),
      this.getPbrPipelineAsync('static', 0, DoubleSide, 0, true, 0, false, s, defaultHdr).then(() => {}),
      this.getPbrPipelineAsync('static', 0, DoubleSide, 0, true, 0, true, s, defaultHdr).then(() => {}),
      this.getPbrPipelineAsync('static', 0, FrontSide, 1, false, 0, false, s, defaultHdr).then(() => {}),
      this.getPbrPipelineAsync('static', 0, DoubleSide, 1, false, 0, false, s, defaultHdr).then(() => {}),
      this.getPbrPipelineAsync('instanced', 1, FrontSide, 0, true, 0, false, s, defaultHdr).then(() => {}),
      this.getPbrPipelineAsync('instanced', 1, DoubleSide, 0, true, 0, false, s, defaultHdr).then(() => {}),
      this.getPbrPipelineAsync('skinned', 2, FrontSide, 0, true, 0, false, s, defaultHdr).then(() => {}),
      this.getShadowPipelineAsync('static', 0, FrontSide).then(() => {}),
      this.getShadowPipelineAsync('static', 0, DoubleSide).then(() => {}),
      this.getShadowPipelineAsync('instanced', 1, FrontSide).then(() => {}),
      this.getShadowPipelineAsync('skinned', 2, FrontSide).then(() => {}),
      this.getSkyPipelineAsync(s, defaultHdr).then(() => {}),
      this.getPointsPipelineAsync(1, s, defaultHdr).then(() => {}),
      this.getPointsPipelineAsync(2, s, defaultHdr).then(() => {}),
      this.getLinesPipelineAsync(1, s, defaultHdr).then(() => {}),
    ];
    await Promise.all(warmTasks);
  }

  private rebuildDrawBoneBg(): void {
    this.drawBoneBg = this.device.createBindGroup({
      layout: this.drawBoneBgl,
      entries: [
        { binding: 0, resource: { buffer: this.drawStorageBuf } },
        { binding: 1, resource: { buffer: this.boneStorageBuf } },
      ],
    });
  }

  private rebuildSceneBg(shadowView: GPUTextureViewLike): void {
    if (this.sceneBgShadowView === shadowView) return;
    this.sceneBgShadowView = shadowView;
    this.sceneBg = this.device.createBindGroup({
      layout: this.sceneBgl,
      entries: [
        { binding: 0, resource: { buffer: this.sceneUniformBuf } },
        { binding: 1, resource: shadowView },
        { binding: 2, resource: this.sampShadowCmp },
      ],
    });
  }

  // =========================================================================
  // Pipeline Factories (Compact Key Cache — ~12-16 Pipelines Total!)
  // =========================================================================

  private pbrVertexBuffers(isSkin: boolean): unknown[] {
    const bufs: unknown[] = [{
      arrayStride: 56, // 14 floats: pos(3) + nor(3) + uv(2) + uv1(2) + col(4)
      stepMode: 'vertex',
      attributes: [
        { shaderLocation: 0, offset: 0,  format: 'float32x3' },
        { shaderLocation: 1, offset: 12, format: 'float32x3' },
        { shaderLocation: 2, offset: 24, format: 'float32x2' },
        { shaderLocation: 3, offset: 32, format: 'float32x2' },
        { shaderLocation: 4, offset: 40, format: 'float32x4' },
      ],
    }];
    if (isSkin) {
      bufs.push({
        arrayStride: 32, // 8 floats: skinIndex(4) + skinWeight(4)
        stepMode: 'vertex',
        attributes: [
          { shaderLocation: 5, offset: 0,  format: 'float32x4' },
          { shaderLocation: 6, offset: 16, format: 'float32x4' },
        ],
      });
    }
    return bufs;
  }

  private makePbrPipelineDesc(
    mode: DrawMode,
    side: number,
    blendMode: number,
    depthWrite: boolean,
    polyKey: number,
    a2c: boolean,
    samples: number,
    isHdr: boolean,
  ): unknown {
    const cullMode = side === DoubleSide ? 'none' : side === BackSide ? 'front' : 'back';
    const target: Record<string, unknown> = { format: isHdr ? this.hdrFormat : this.canvasFormat };
    if (blendMode === 1) {
      target.blend = {
        color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      };
    } else if (blendMode === 2) {
      target.blend = {
        color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
        alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
      };
    }
    const polyFactor = polyKey ? -2 : 0;
    return {
      layout: this.pbrLayouts[mode],
      vertex: {
        module: this.pbrMods[mode],
        entryPoint: 'vsMain',
        buffers: this.pbrVertexBuffers(mode === 'skinned'),
      },
      fragment: {
        module: this.pbrMods[mode],
        entryPoint: 'fsMain',
        targets: [target],
      },
      primitive: { topology: 'triangle-list', frontFace: 'ccw', cullMode },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: depthWrite,
        depthCompare: 'less-equal',
        depthBias: polyFactor,
        depthBiasSlopeScale: polyFactor,
      },
      multisample: {
        count: samples,
        alphaToCoverageEnabled: a2c && samples > 1,
      },
    };
  }

  private pbrPipeKey(modeId: number, side: number, blendMode: number, depthWrite: boolean, polyKey: number, a2c: boolean, samples: number, isHdr: boolean): number {
    return (modeId << 14) | (side << 12) | (blendMode << 10) | ((depthWrite ? 1 : 0) << 9) | (polyKey << 5) | ((a2c && samples > 1 ? 1 : 0) << 4) | ((samples > 1 ? 1 : 0) << 1) | (isHdr ? 1 : 0);
  }

  private async getPbrPipelineAsync(mode: DrawMode, modeId: number, side: number, blendMode: number, depthWrite: boolean, polyKey: number, a2c: boolean, samples: number, isHdr: boolean): Promise<GPURenderPipelineLike> {
    const key = this.pbrPipeKey(modeId, side, blendMode, depthWrite, polyKey, a2c, samples, isHdr);
    const existing = this.pbrPipeCache.get(key);
    if (existing) return existing;
    const pipe = await this.device.createRenderPipelineAsync(this.makePbrPipelineDesc(mode, side, blendMode, depthWrite, polyKey, a2c, samples, isHdr));
    this.pbrPipeCache.set(key, pipe);
    this.info.programs.push(pipe);
    return pipe;
  }

  private getPbrPipeline(mode: DrawMode, modeId: number, side: number, blendMode: number, depthWrite: boolean, polyKey: number, a2c: boolean, samples: number, isHdr: boolean): GPURenderPipelineLike {
    const key = this.pbrPipeKey(modeId, side, blendMode, depthWrite, polyKey, a2c, samples, isHdr);
    let pipe = this.pbrPipeCache.get(key);
    if (!pipe) {
      pipe = this.device.createRenderPipeline(this.makePbrPipelineDesc(mode, side, blendMode, depthWrite, polyKey, a2c, samples, isHdr));
      this.pbrPipeCache.set(key, pipe);
      this.info.programs.push(pipe);
    }
    return pipe;
  }

  private makeShadowPipelineDesc(mode: DrawMode, side: number): unknown {
    return {
      layout: this.shadowLayouts[mode],
      vertex: {
        module: this.shadowMods[mode],
        entryPoint: 'vsMain',
        buffers: this.pbrVertexBuffers(mode === 'skinned'),
      },
      fragment: {
        module: this.shadowMods[mode],
        entryPoint: 'fsMain',
        targets: [],
      },
      primitive: {
        topology: 'triangle-list',
        frontFace: 'ccw',
        cullMode: side === DoubleSide ? 'none' : 'back',
      },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: true,
        depthCompare: 'less-equal',
        depthBias: 3,
        depthBiasSlopeScale: 2.2,
      },
    };
  }

  private async getShadowPipelineAsync(mode: DrawMode, modeId: number, side: number): Promise<GPURenderPipelineLike> {
    const key = (modeId << 2) | (side === DoubleSide ? 1 : 0);
    const existing = this.shadowPipeCache.get(key);
    if (existing) return existing;
    const pipe = await this.device.createRenderPipelineAsync(this.makeShadowPipelineDesc(mode, side));
    this.shadowPipeCache.set(key, pipe);
    this.info.programs.push(pipe);
    return pipe;
  }

  private getShadowPipeline(mode: DrawMode, modeId: number, side: number): GPURenderPipelineLike {
    const key = (modeId << 2) | (side === DoubleSide ? 1 : 0);
    let pipe = this.shadowPipeCache.get(key);
    if (!pipe) {
      pipe = this.device.createRenderPipeline(this.makeShadowPipelineDesc(mode, side));
      this.shadowPipeCache.set(key, pipe);
      this.info.programs.push(pipe);
    }
    return pipe;
  }

  private makeSkyPipelineDesc(samples: number, isHdr: boolean): unknown {
    return {
      layout: this.skyLayout,
      vertex: { module: this.skyMod, entryPoint: 'vsMain', buffers: this.pbrVertexBuffers(false) },
      fragment: { module: this.skyMod, entryPoint: 'fsMain', targets: [{ format: isHdr ? this.hdrFormat : this.canvasFormat }] },
      primitive: { topology: 'triangle-list', frontFace: 'ccw', cullMode: 'front' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less-equal' },
      multisample: { count: samples },
    };
  }

  private async getSkyPipelineAsync(samples: number, isHdr: boolean): Promise<GPURenderPipelineLike> {
    const key = (samples << 1) | (isHdr ? 1 : 0);
    const existing = this.skyPipeCache.get(key);
    if (existing) return existing;
    const pipe = await this.device.createRenderPipelineAsync(this.makeSkyPipelineDesc(samples, isHdr));
    this.skyPipeCache.set(key, pipe);
    this.info.programs.push(pipe);
    return pipe;
  }

  private getSkyPipeline(samples: number, isHdr: boolean): GPURenderPipelineLike {
    const key = (samples << 1) | (isHdr ? 1 : 0);
    let pipe = this.skyPipeCache.get(key);
    if (!pipe) {
      pipe = this.device.createRenderPipeline(this.makeSkyPipelineDesc(samples, isHdr));
      this.skyPipeCache.set(key, pipe);
      this.info.programs.push(pipe);
    }
    return pipe;
  }

  private makePointsPipelineDesc(blendMode: number, samples: number, isHdr: boolean): unknown {
    const target: Record<string, unknown> = { format: isHdr ? this.hdrFormat : this.canvasFormat };
    if (blendMode === 1) {
      target.blend = {
        color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      };
    } else if (blendMode === 2) {
      target.blend = {
        color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
        alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
      };
    }
    return {
      layout: this.pointsLayout,
      vertex: {
        module: this.pointsMod,
        entryPoint: 'vsMain',
        buffers: [{
          arrayStride: 32,
          stepMode: 'instance',
          attributes: [
            { shaderLocation: 0, offset: 0,  format: 'float32x3' },
            { shaderLocation: 1, offset: 12, format: 'float32' },
            { shaderLocation: 2, offset: 16, format: 'float32x3' },
            { shaderLocation: 3, offset: 28, format: 'float32' },
          ],
        }],
      },
      fragment: { module: this.pointsMod, entryPoint: 'fsMain', targets: [target] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less-equal' },
      multisample: { count: samples },
    };
  }

  private async getPointsPipelineAsync(blendMode: number, samples: number, isHdr: boolean): Promise<GPURenderPipelineLike> {
    const key = (blendMode << 4) | (samples << 1) | (isHdr ? 1 : 0);
    const existing = this.pointsPipeCache.get(key);
    if (existing) return existing;
    const pipe = await this.device.createRenderPipelineAsync(this.makePointsPipelineDesc(blendMode, samples, isHdr));
    this.pointsPipeCache.set(key, pipe);
    this.info.programs.push(pipe);
    return pipe;
  }

  private getPointsPipeline(blendMode: number, samples: number, isHdr: boolean): GPURenderPipelineLike {
    const key = (blendMode << 4) | (samples << 1) | (isHdr ? 1 : 0);
    let pipe = this.pointsPipeCache.get(key);
    if (!pipe) {
      pipe = this.device.createRenderPipeline(this.makePointsPipelineDesc(blendMode, samples, isHdr));
      this.pointsPipeCache.set(key, pipe);
      this.info.programs.push(pipe);
    }
    return pipe;
  }

  private makeLinesPipelineDesc(blendMode: number, samples: number, isHdr: boolean): unknown {
    const target: Record<string, unknown> = { format: isHdr ? this.hdrFormat : this.canvasFormat };
    if (blendMode === 1) {
      target.blend = {
        color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      };
    } else if (blendMode === 2) {
      target.blend = {
        color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
        alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
      };
    }
    return {
      layout: this.linesLayout,
      vertex: {
        module: this.linesMod,
        entryPoint: 'vsMain',
        buffers: [{
          arrayStride: 12,
          stepMode: 'vertex',
          attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }],
        }],
      },
      fragment: { module: this.linesMod, entryPoint: 'fsMain', targets: [target] },
      primitive: { topology: 'line-list' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less-equal' },
      multisample: { count: samples },
    };
  }

  private async getLinesPipelineAsync(blendMode: number, samples: number, isHdr: boolean): Promise<GPURenderPipelineLike> {
    const key = (blendMode << 4) | (samples << 1) | (isHdr ? 1 : 0);
    const existing = this.linesPipeCache.get(key);
    if (existing) return existing;
    const pipe = await this.device.createRenderPipelineAsync(this.makeLinesPipelineDesc(blendMode, samples, isHdr));
    this.linesPipeCache.set(key, pipe);
    this.info.programs.push(pipe);
    return pipe;
  }

  private getLinesPipeline(blendMode: number, samples: number, isHdr: boolean): GPURenderPipelineLike {
    const key = (blendMode << 4) | (samples << 1) | (isHdr ? 1 : 0);
    let pipe = this.linesPipeCache.get(key);
    if (!pipe) {
      pipe = this.device.createRenderPipeline(this.makeLinesPipelineDesc(blendMode, samples, isHdr));
      this.linesPipeCache.set(key, pipe);
      this.info.programs.push(pipe);
    }
    return pipe;
  }

  // =========================================================================
  // Texture Sync (Gamma-Aware GPU Mipmaps + Cached Material Bind Groups)
  // =========================================================================

  private destroyTextureGpu(gpuTex: GPUTextureLike): void {
    for (let i = this.pendingMips.length - 1; i >= 0; i--) {
      if (this.pendingMips[i].tex === gpuTex) this.pendingMips.splice(i, 1);
    }
    gpuTex.destroy();
  }

  private flushPendingMipmaps(enc: GPUCommandEncoderLike): void {
    if (this.pendingMips.length === 0) return;
    const dev = this.device;
    for (let i = 0; i < this.pendingMips.length; i++) {
      const { tex, mipCount, isSrgb } = this.pendingMips[i];
      const pipe = isSrgb ? this.mipSrgbPipe : this.mipLinPipe;
      const readFmt = isSrgb ? 'rgba8unorm-srgb' : 'rgba8unorm';
      for (let lvl = 1; lvl < mipCount; lvl++) {
        const srcView = tex.createView({ format: readFmt, baseMipLevel: lvl - 1, mipLevelCount: 1 });
        const dstView = tex.createView({ format: 'rgba8unorm', baseMipLevel: lvl, mipLevelCount: 1 });
        const bg = dev.createBindGroup({
          layout: this.mipBgl,
          entries: [
            { binding: 0, resource: srcView },
            { binding: 1, resource: this.sampClampLin },
          ],
        });
        const pass = enc.beginRenderPass({
          colorAttachments: [{ view: dstView, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 } }],
        });
        pass.setPipeline(pipe);
        pass.setBindGroup(0, bg);
        pass.draw(3);
        pass.end();
      }
    }
    this.pendingMips.length = 0;
  }

  private toCanvasSource(img: TexImageSource, w: number, h: number): HTMLCanvasElement {
    if (img instanceof HTMLCanvasElement) return img;
    if (!this.scratchCanvas) this.scratchCanvas = document.createElement('canvas');
    const c = this.scratchCanvas;
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    const ctx = c.getContext('2d')!;
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(img as CanvasImageSource, 0, 0, w, h);
    return c;
  }

  private syncTexture(tex: Texture | null, fallback: GPUTextureViewLike): GPUTextureViewLike {
    if (!tex) return fallback;
    const dev = this.device;
    let rec = this.texCache.get(tex.id);
    if (rec && rec.disposedVersion !== tex._disposedVersion) {
      this.destroyTextureGpu(rec.tex);
      this.texCache.delete(tex.id);
      rec = undefined;
    }

    const needUpload = !rec || rec.version !== tex.version;
    if (!needUpload && rec) return rec.view;

    const isSrgb = tex.colorSpace === SRGBColorSpace;
    const viewFmt = isSrgb ? 'rgba8unorm-srgb' : 'rgba8unorm';
    const wrapRepeat = tex.wrapS === RepeatWrapping;

    if (tex.isDataTexture && tex.data) {
      const w = tex.width, h = tex.height;
      if (w <= 0 || h <= 0) return fallback;
      if (rec) this.destroyTextureGpu(rec.tex);
      const gpuTex = dev.createTexture({
        size: [w, h],
        mipLevelCount: 1,
        format: 'rgba8unorm',
        viewFormats: isSrgb ? ['rgba8unorm-srgb'] : [],
        usage: GPU_TEXTURE_USAGE.TEXTURE_BINDING | GPU_TEXTURE_USAGE.COPY_DST,
      });
      let data = tex.data;
      if (tex.flipY && h > 1) {
        const flipped = new Uint8Array(w * h * 4);
        const rowBytes = w * 4;
        for (let y = 0; y < h; y++) {
          flipped.set(data.subarray((h - 1 - y) * rowBytes, (h - y) * rowBytes), y * rowBytes);
        }
        data = flipped;
      }
      dev.queue.writeTexture({ texture: gpuTex }, data, { bytesPerRow: w * 4, rowsPerImage: h }, [w, h]);
      const view = gpuTex.createView({ format: viewFmt });
      rec = { tex: gpuTex, view, version: tex.version, disposedVersion: tex._disposedVersion, isSrgb, wrapRepeat, aniso: tex.anisotropy };
    } else if (tex.image) {
      const img = tex.image as { width?: number; height?: number; naturalWidth?: number; naturalHeight?: number };
      const w = img.naturalWidth || img.width || 0;
      const h = img.naturalHeight || img.height || 0;
      if (w <= 0 || h <= 0) return fallback;
      if (rec) this.destroyTextureGpu(rec.tex);
      const mipCount = Math.floor(Math.log2(Math.max(w, h))) + 1;
      const gpuTex = dev.createTexture({
        size: [w, h],
        mipLevelCount: mipCount,
        format: 'rgba8unorm',
        viewFormats: isSrgb ? ['rgba8unorm-srgb'] : [],
        usage: GPU_TEXTURE_USAGE.TEXTURE_BINDING | GPU_TEXTURE_USAGE.COPY_DST | GPU_TEXTURE_USAGE.RENDER_ATTACHMENT,
      });
      const srcCanvas = this.toCanvasSource(tex.image as TexImageSource, w, h);
      dev.queue.copyExternalImageToTexture(
        { source: srcCanvas, flipY: tex.flipY },
        { texture: gpuTex, mipLevel: 0 },
        [w, h],
      );
      if (mipCount > 1) {
        this.pendingMips.push({ tex: gpuTex, mipCount, isSrgb });
      }
      const view = gpuTex.createView({ format: viewFmt });
      rec = { tex: gpuTex, view, version: tex.version, disposedVersion: tex._disposedVersion, isSrgb, wrapRepeat, aniso: tex.anisotropy };
    } else {
      return fallback;
    }

    this.texCache.set(tex.id, rec);
    tex._onDispose = (t) => {
      const r = this.texCache.get(t.id);
      if (r) {
        this.destroyTextureGpu(r.tex);
        this.texCache.delete(t.id);
        this.matBgCache.clear();
        this.info.memory.textures = this.texCache.size;
      }
    };
    this.info.memory.textures = this.texCache.size;
    return rec.view;
  }

  private getMaterialBindGroup(mat: Material): GPUBindGroupLike {
    const mapTex = mat.map;
    const emTex = mat.emissiveMap;
    const crmTex = mat.roughnessMap ?? mat.metalnessMap ?? mat.clearcoatMap;
    const detailTex = mat.bumpMap ?? mat.aoMap;
    const maskTex = mat.personOutfit?.mask ?? null;
    const overTex = mat.personOutfit?.overlay ?? null;

    const mapView = this.syncTexture(mapTex, this.whiteView);
    const emView = this.syncTexture(emTex, this.blackView);
    const crmView = this.syncTexture(crmTex, this.whiteView);
    const detailView = this.syncTexture(detailTex, this.whiteView);
    const maskView = this.syncTexture(maskTex, this.blackView);
    const overView = this.syncTexture(overTex, this.clearView);

    const k = `${mapTex ? `${mapTex.id}:${mapTex.version}:${mapTex._disposedVersion}:${mapTex.wrapS}:${mapTex.isDataTexture ? 1 : 0}` : 0}|${emTex ? `${emTex.id}:${emTex.version}:${emTex._disposedVersion}` : 0}|${crmTex ? `${crmTex.id}:${crmTex.version}:${crmTex._disposedVersion}` : 0}|${detailTex ? `${detailTex.id}:${detailTex.version}:${detailTex._disposedVersion}` : 0}|${maskTex ? `${maskTex.id}:${maskTex.version}:${maskTex._disposedVersion}` : 0}|${overTex ? `${overTex.id}:${overTex.version}:${overTex._disposedVersion}` : 0}`;
    const cached = this.matBgCache.get(mat.id);
    if (cached && cached.key === k) return cached.bg;

    const mapSampler = mapTex
      ? (mapTex.isDataTexture
        ? this.sampNearestClamp
        : (mapTex.wrapS === RepeatWrapping ? this.sampRepeatAniso : this.sampClampAniso))
      : this.sampRepeatLin;

    const bg = this.device.createBindGroup({
      layout: this.matBgl,
      entries: [
        { binding: 0, resource: mapView },
        { binding: 1, resource: mapSampler },
        { binding: 2, resource: emView },
        { binding: 3, resource: crmView },
        { binding: 4, resource: detailView },
        { binding: 5, resource: maskView },
        { binding: 6, resource: overView },
        { binding: 7, resource: this.sampRepeatLin },
        { binding: 8, resource: this.sampClampLin },
      ],
    });
    this.matBgCache.set(mat.id, { bg, key: k });
    return bg;
  }

  // =========================================================================
  // Geometry Sync (Interleaved 56-byte Vertex Buffer for Optimal GPU Cache!)
  // =========================================================================

  private packInterleavedVerts(geo: BufferGeometry): Float32Array {
    const posAttr = geo.attributes.position;
    const count = posAttr.count;
    const out = new Float32Array(count * 14);
    const pos = posAttr.array;
    const posN = attrNormScale(posAttr);
    const norAttr = geo.attributes.normal;
    const nor = norAttr?.array;
    const norN = attrNormScale(norAttr);
    const uvAttr = geo.attributes.uv;
    const uv = uvAttr?.array;
    const uvN = attrNormScale(uvAttr);
    const uv1Attr = geo.attributes.uv1;
    const uv1 = uv1Attr?.array;
    const uv1N = attrNormScale(uv1Attr);
    const colAttr = geo.attributes.color;
    const col = colAttr?.array;
    const colSize = colAttr?.itemSize ?? 0;
    const colN = attrNormScale(colAttr);

    const pS = posN.scale, pSg = posN.signed;
    const nS = norN.scale, nSg = norN.signed;
    const uS = uvN.scale,  uSg = uvN.signed;
    const u1S = uv1N.scale, u1Sg = uv1N.signed;
    const cS = colN.scale, cSg = colN.signed;

    for (let i = 0; i < count; i++) {
      const o = i * 14;
      const i3 = i * 3;
      const i2 = i * 2;
      if (pS === 1) {
        out[o] = pos[i3]; out[o + 1] = pos[i3 + 1]; out[o + 2] = pos[i3 + 2];
      } else if (pSg) {
        out[o] = Math.max(-1, pos[i3] * pS);
        out[o + 1] = Math.max(-1, pos[i3 + 1] * pS);
        out[o + 2] = Math.max(-1, pos[i3 + 2] * pS);
      } else {
        out[o] = pos[i3] * pS; out[o + 1] = pos[i3 + 1] * pS; out[o + 2] = pos[i3 + 2] * pS;
      }

      if (nor) {
        if (nS === 1) {
          out[o + 3] = nor[i3]; out[o + 4] = nor[i3 + 1]; out[o + 5] = nor[i3 + 2];
        } else if (nSg) {
          out[o + 3] = Math.max(-1, nor[i3] * nS);
          out[o + 4] = Math.max(-1, nor[i3 + 1] * nS);
          out[o + 5] = Math.max(-1, nor[i3 + 2] * nS);
        } else {
          out[o + 3] = nor[i3] * nS; out[o + 4] = nor[i3 + 1] * nS; out[o + 5] = nor[i3 + 2] * nS;
        }
      } else {
        out[o + 3] = 0; out[o + 4] = 1; out[o + 5] = 0;
      }

      if (uv) {
        out[o + 6] = uSg ? Math.max(-1, uv[i2] * uS) : uv[i2] * uS;
        out[o + 7] = uSg ? Math.max(-1, uv[i2 + 1] * uS) : uv[i2 + 1] * uS;
      }
      if (uv1) {
        out[o + 8] = u1Sg ? Math.max(-1, uv1[i2] * u1S) : uv1[i2] * u1S;
        out[o + 9] = u1Sg ? Math.max(-1, uv1[i2 + 1] * u1S) : uv1[i2 + 1] * u1S;
      }
      if (col && colSize >= 3) {
        const ci = i * colSize;
        if (cSg) {
          out[o + 10] = Math.max(-1, col[ci] * cS);
          out[o + 11] = Math.max(-1, col[ci + 1] * cS);
          out[o + 12] = Math.max(-1, col[ci + 2] * cS);
          out[o + 13] = colSize >= 4 ? Math.max(-1, col[ci + 3] * cS) : 1;
        } else {
          out[o + 10] = col[ci] * cS;
          out[o + 11] = col[ci + 1] * cS;
          out[o + 12] = col[ci + 2] * cS;
          out[o + 13] = colSize >= 4 ? col[ci + 3] * cS : 1;
        }
      } else {
        out[o + 10] = 1; out[o + 11] = 1; out[o + 12] = 1; out[o + 13] = 1;
      }
    }
    return out;
  }

  private packSkinVerts(geo: BufferGeometry): Float32Array | null {
    const siAttr = geo.attributes.skinIndex;
    const swAttr = geo.attributes.skinWeight;
    if (!siAttr || !swAttr) return null;
    const count = geo.attributes.position.count;
    const out = new Float32Array(count * 8);
    const si = siAttr.array, sw = swAttr.array;
    const swN = attrNormScale(swAttr);
    const wS = swN.scale, wSg = swN.signed;
    for (let i = 0; i < count; i++) {
      const o = i * 8, i4 = i * 4;
      out[o] = si[i4]; out[o + 1] = si[i4 + 1]; out[o + 2] = si[i4 + 2]; out[o + 3] = si[i4 + 3];
      if (wSg) {
        out[o + 4] = Math.max(-1, sw[i4] * wS);
        out[o + 5] = Math.max(-1, sw[i4 + 1] * wS);
        out[o + 6] = Math.max(-1, sw[i4 + 2] * wS);
        out[o + 7] = Math.max(-1, sw[i4 + 3] * wS);
      } else {
        out[o + 4] = sw[i4] * wS;
        out[o + 5] = sw[i4 + 1] * wS;
        out[o + 6] = sw[i4 + 2] * wS;
        out[o + 7] = sw[i4 + 3] * wS;
      }
    }
    return out;
  }

  private deleteGeoGpu(rec: GeoGpu): void {
    rec.vbuf.destroy();
    if (rec.skinBuf) rec.skinBuf.destroy();
    if (rec.ibuf) rec.ibuf.destroy();
  }

  private syncGeometry(geo: BufferGeometry): GeoGpu {
    const dev = this.device;
    let rec = this.geoCache.get(geo.id);
    if (rec && rec.disposedVersion !== geo._disposedVersion) {
      this.deleteGeoGpu(rec);
      this.geoCache.delete(geo.id);
      rec = undefined;
    }

    let dirty = !rec || rec.layoutVersion !== geo._layoutVersion || (geo.index ? rec.indexVersion !== geo.index.version : rec.ibuf !== null);
    if (!dirty && rec) {
      for (const name of ['position', 'normal', 'uv', 'uv1', 'color', 'skinIndex', 'skinWeight']) {
        const a: BufferAttribute | undefined = geo.attributes[name];
        const prev = rec.attrVersions.get(name);
        if ((a ? a.version : -1) !== (prev ?? -1)) {
          dirty = true;
          break;
        }
      }
    }

    if (!dirty && rec) return rec;

    if (rec) this.deleteGeoGpu(rec);

    const packed = this.packInterleavedVerts(geo);
    const vbuf = dev.createBuffer({
      size: Math.max(56, packed.byteLength),
      usage: GPU_BUFFER_USAGE.VERTEX | GPU_BUFFER_USAGE.COPY_DST,
    });
    if (packed.byteLength > 0) dev.queue.writeBuffer(vbuf, 0, packed);

    const skinPacked = this.packSkinVerts(geo);
    let skinBuf: GPUBufferLike | null = null;
    if (skinPacked) {
      skinBuf = dev.createBuffer({
        size: Math.max(32, skinPacked.byteLength),
        usage: GPU_BUFFER_USAGE.VERTEX | GPU_BUFFER_USAGE.COPY_DST,
      });
      dev.queue.writeBuffer(skinBuf, 0, skinPacked);
    }

    let ibuf: GPUBufferLike | null = null;
    let indexFormat: 'uint16' | 'uint32' = 'uint16';
    let indexCount = 0;
    let indexVersion = -1;
    if (geo.index) {
      const idxArr = geo.index.array;
      indexFormat = idxArr instanceof Uint32Array ? 'uint32' : 'uint16';
      indexCount = geo.index.count;
      indexVersion = geo.index.version;
      const padded = paddedView4(idxArr);
      ibuf = dev.createBuffer({
        size: Math.max(4, padded.byteLength),
        usage: GPU_BUFFER_USAGE.INDEX | GPU_BUFFER_USAGE.COPY_DST,
      });
      dev.queue.writeBuffer(ibuf, 0, padded);
    }

    const attrVersions = new Map<string, number>();
    for (const name of ['position', 'normal', 'uv', 'uv1', 'color', 'skinIndex', 'skinWeight']) {
      const a = geo.attributes[name];
      attrVersions.set(name, a ? a.version : -1);
    }

    rec = {
      vbuf,
      skinBuf,
      ibuf,
      indexFormat,
      indexCount,
      vertexCount: geo.attributes.position.count,
      layoutVersion: geo._layoutVersion,
      disposedVersion: geo._disposedVersion,
      attrVersions,
      indexVersion,
    };
    this.geoCache.set(geo.id, rec);
    geo._onDispose = (g) => {
      const r = this.geoCache.get(g.id);
      if (r) {
        this.deleteGeoGpu(r);
        this.geoCache.delete(g.id);
        this.info.memory.geometries = this.geoCache.size;
      }
    };
    this.info.memory.geometries = this.geoCache.size;
    return rec;
  }

  // =========================================================================
  // InstancedMesh GPU Buffer Sync (Compute Frustum Culling + Indirect Draw!)
  // =========================================================================

  private destroyInstGpu(e: InstGpu): void {
    e.matBuf.destroy();
    e.colBuf.destroy();
    e.sphereBuf.destroy();
    e.visBuf.destroy();
    e.indirectBuf.destroy();
    e.indirectTemplate.destroy();
    e.slotUniformBuf.destroy();
  }

  private syncInstanced(mesh: InstancedMesh, geoRec: GeoGpu): InstGpu {
    const dev = this.device;
    const total = Math.max(1, mesh.count);
    let e = this.instCache.get(mesh.id);
    if (e && (e.total !== total || e.geoId !== mesh.geometry.id || e.geoDisp !== mesh.geometry._disposedVersion || e.geoLayout !== mesh.geometry._layoutVersion)) {
      this.destroyInstGpu(e);
      this.instCache.delete(mesh.id);
      e = undefined;
    }

    if (!e) {
      const isIndexed = geoRec.indexCount > 0;
      const drawCount = isIndexed ? geoRec.indexCount : geoRec.vertexCount;
      const matBuf = dev.createBuffer({ size: total * 64, usage: GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_DST });
      const colBuf = dev.createBuffer({ size: total * 16, usage: GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_DST });
      const sphereBuf = dev.createBuffer({ size: total * 16, usage: GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_DST });
      // 5 view-slots (slot 0 = main camera, slots 1..4 = shadow cascades 0..3)
      const visBuf = dev.createBuffer({ size: 5 * total * 4, usage: GPU_BUFFER_USAGE.STORAGE });
      const indirectBuf = dev.createBuffer({
        size: 5 * 20,
        usage: GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.INDIRECT | GPU_BUFFER_USAGE.COPY_DST,
      });
      const indirectTemplate = dev.createBuffer({
        size: 5 * 20,
        usage: GPU_BUFFER_USAGE.COPY_SRC | GPU_BUFFER_USAGE.COPY_DST,
      });
      const tmpl = new Uint32Array(5 * 5);
      for (let s = 0; s < 5; s++) {
        tmpl[s * 5 + 0] = drawCount; // indexCount or vertexCount
        tmpl[s * 5 + 1] = 0;         // instanceCount (atomically incremented by Compute Shader!)
        tmpl[s * 5 + 2] = 0;         // firstIndex / firstVertex
        tmpl[s * 5 + 3] = 0;         // baseVertex / firstInstance
        tmpl[s * 5 + 4] = 0;         // firstInstance (indexed) / pad
      }
      dev.queue.writeBuffer(indirectTemplate, 0, tmpl);

      const slotUniformBuf = dev.createBuffer({
        size: 5 * this.uAlign,
        usage: GPU_BUFFER_USAGE.UNIFORM | GPU_BUFFER_USAGE.COPY_DST,
      });

      const renderBg = dev.createBindGroup({
        layout: this.instBgl,
        entries: [
          { binding: 0, resource: { buffer: matBuf } },
          { binding: 1, resource: { buffer: colBuf } },
          { binding: 2, resource: { buffer: visBuf } },
          { binding: 3, resource: { buffer: slotUniformBuf, offset: 0, size: 16 } },
        ],
      });
      const cullBg = dev.createBindGroup({
        layout: this.cullBgl,
        entries: [
          { binding: 0, resource: { buffer: this.cullParamsBuf, offset: 0, size: 112 } },
          { binding: 1, resource: { buffer: sphereBuf } },
          { binding: 2, resource: { buffer: visBuf } },
          { binding: 3, resource: { buffer: indirectBuf } },
        ],
      });

      const colsCpu = new Float32Array(total * 4);
      colsCpu.fill(1);
      dev.queue.writeBuffer(colBuf, 0, colsCpu);

      e = {
        total,
        geoId: mesh.geometry.id,
        geoDisp: mesh.geometry._disposedVersion,
        geoLayout: mesh.geometry._layoutVersion,
        matBuf,
        colBuf,
        sphereBuf,
        visBuf,
        indirectBuf,
        indirectTemplate,
        slotUniformBuf,
        renderBg,
        cullBg,
        matVer: -1,
        colVer: -1,
        worldMat: new Float32Array(12),
        worldMatValid: false,
        isIndexed,
        drawCount,
        spheresCpu: new Float32Array(total * 4),
        colsCpu,
        slotCpu: new Uint32Array((5 * this.uAlign) / 4),
      };
      this.instCache.set(mesh.id, e);
      mesh._onDispose = (im) => {
        const rec = this.instCache.get(im.id);
        if (rec) {
          this.destroyInstGpu(rec);
          this.instCache.delete(im.id);
        }
      };
    }

    const mw = mesh.matrixWorld.elements;
    const matDirty = e.matVer !== mesh.instanceMatrix.version;
    if (matDirty) {
      dev.queue.writeBuffer(e.matBuf, 0, mesh.instanceMatrix.array, 0, mesh.count * 16);
      e.matVer = mesh.instanceMatrix.version;
    }
    const wm = e.worldMat;
    const worldDirty = !e.worldMatValid
      || wm[0] !== mw[0] || wm[1] !== mw[1] || wm[2] !== mw[2]
      || wm[3] !== mw[4] || wm[4] !== mw[5] || wm[5] !== mw[6]
      || wm[6] !== mw[8] || wm[7] !== mw[9] || wm[8] !== mw[10]
      || wm[9] !== mw[12] || wm[10] !== mw[13] || wm[11] !== mw[14];
    if (matDirty || worldDirty) {
      const bs = mesh.geometry.boundingSphere ?? mesh.geometry.computeBoundingSphere();
      const bx = bs.center.x, by = bs.center.y, bz = bs.center.z, br = bs.radius;
      const src = mesh.instanceMatrix.array;
      const sph = e.spheresCpu;
      const wSc = Math.sqrt(Math.max(
        mw[0] * mw[0] + mw[1] * mw[1] + mw[2] * mw[2],
        mw[4] * mw[4] + mw[5] * mw[5] + mw[6] * mw[6],
        mw[8] * mw[8] + mw[9] * mw[9] + mw[10] * mw[10],
      ));
      for (let i = 0; i < mesh.count; i++) {
        const o = i * 16;
        const m0 = src[o], m1 = src[o + 1], m2 = src[o + 2];
        const m4 = src[o + 4], m5 = src[o + 5], m6 = src[o + 6];
        const m8 = src[o + 8], m9 = src[o + 9], m10 = src[o + 10];
        const lx = m0 * bx + m4 * by + m8 * bz + src[o + 12];
        const ly = m1 * bx + m5 * by + m9 * bz + src[o + 13];
        const lz = m2 * bx + m6 * by + m10 * bz + src[o + 14];
        const s = i * 4;
        sph[s]     = mw[0] * lx + mw[4] * ly + mw[8] * lz + mw[12];
        sph[s + 1] = mw[1] * lx + mw[5] * ly + mw[9] * lz + mw[13];
        sph[s + 2] = mw[2] * lx + mw[6] * ly + mw[10] * lz + mw[14];
        const sc = Math.sqrt(Math.max(m0 * m0 + m1 * m1 + m2 * m2, m4 * m4 + m5 * m5 + m6 * m6, m8 * m8 + m9 * m9 + m10 * m10));
        sph[s + 3] = br * sc * wSc + 0.5;
      }
      dev.queue.writeBuffer(e.sphereBuf, 0, sph, 0, mesh.count * 4);
      wm[0] = mw[0]; wm[1] = mw[1]; wm[2] = mw[2];
      wm[3] = mw[4]; wm[4] = mw[5]; wm[5] = mw[6];
      wm[6] = mw[8]; wm[7] = mw[9]; wm[8] = mw[10];
      wm[9] = mw[12]; wm[10] = mw[13]; wm[11] = mw[14];
      e.worldMatValid = true;
    }

    if (mesh.instanceColor && e.colVer !== mesh.instanceColor.version) {
      const srcC = mesh.instanceColor.array;
      const dstC = e.colsCpu;
      for (let i = 0; i < mesh.count; i++) {
        dstC[i * 4] = srcC[i * 3];
        dstC[i * 4 + 1] = srcC[i * 3 + 1];
        dstC[i * 4 + 2] = srcC[i * 3 + 2];
        dstC[i * 4 + 3] = 1;
      }
      dev.queue.writeBuffer(e.colBuf, 0, dstC, 0, mesh.count * 4);
      e.colVer = mesh.instanceColor.version;
    }

    return e;
  }

  // =========================================================================
  // Shadow & Post-Process Target Management
  // =========================================================================

  private ensureShadowTargets(size: number, layers: number): void {
    if (this.shadowArrayTex && this.shadowTexSize === size && this.shadowLayers === layers) return;
    if (this.shadowArrayTex) this.shadowArrayTex.destroy();
    const tex = this.device.createTexture({
      size: [size, size, layers],
      format: 'depth24plus',
      usage: GPU_TEXTURE_USAGE.RENDER_ATTACHMENT | GPU_TEXTURE_USAGE.TEXTURE_BINDING,
    });
    this.shadowArrayTex = tex;
    this.shadowArrayView = tex.createView({ dimension: '2d-array', baseArrayLayer: 0, arrayLayerCount: layers });
    this.shadowLayerViews = [];
    for (let c = 0; c < layers; c++) {
      this.shadowLayerViews.push(tex.createView({ dimension: '2d', baseArrayLayer: c, arrayLayerCount: 1 }));
    }
    this.shadowTexSize = size;
    this.shadowLayers = layers;
  }

  private freeRenderTargets(): void {
    if (this.msaaColorTex) { this.msaaColorTex.destroy(); this.msaaColorTex = null; this.msaaColorView = null; }
    if (this.depthTex) { this.depthTex.destroy(); this.depthTex = null; this.depthView = null; }
    if (this.sceneColorTex) { this.sceneColorTex.destroy(); this.sceneColorTex = null; this.sceneColorView = null; }
    if (this.aoTex) { this.aoTex.destroy(); this.aoTex = null; this.aoSampleView = null; this.aoStorageView = null; }
    if (this.bloomTexA) { this.bloomTexA.destroy(); this.bloomTexA = null; this.bloomViewA = null; }
    if (this.bloomTexB) { this.bloomTexB.destroy(); this.bloomTexB = null; this.bloomViewB = null; }
    this.ssaoBg = this.bloomExtBg = this.bloomBlurHBg = this.bloomBlurVBg = this.compBg = null;
    this.tgtW = this.tgtH = this.tgtAoW = this.tgtAoH = 0;
  }

  private ensureRenderTargets(w: number, h: number, usePost: boolean, aoScale: number): void {
    const aoW = usePost && aoScale > 0 ? Math.max(1, Math.round(w * aoScale)) : 0;
    const aoH = usePost && aoScale > 0 ? Math.max(1, Math.round(h * aoScale)) : 0;
    const samples = this.msaaSamples;
    if (this.depthTex && this.tgtW === w && this.tgtH === h && this.tgtHdr === usePost && this.tgtMsaa === samples && this.tgtAoW === aoW && this.tgtAoH === aoH) {
      return;
    }

    this.freeRenderTargets();
    const dev = this.device;

    const colorFmt = usePost ? this.hdrFormat : this.canvasFormat;
    if (samples > 1) {
      this.msaaColorTex = dev.createTexture({
        size: [w, h],
        sampleCount: samples,
        format: colorFmt,
        usage: GPU_TEXTURE_USAGE.RENDER_ATTACHMENT,
      });
      this.msaaColorView = this.msaaColorTex.createView();
    }

    this.depthTex = dev.createTexture({
      size: [w, h],
      sampleCount: samples,
      format: 'depth24plus',
      usage: GPU_TEXTURE_USAGE.RENDER_ATTACHMENT | (aoW > 0 ? GPU_TEXTURE_USAGE.TEXTURE_BINDING : 0),
    });
    this.depthView = this.depthTex.createView();

    if (usePost) {
      this.sceneColorTex = dev.createTexture({
        size: [w, h],
        format: this.hdrFormat,
        usage: GPU_TEXTURE_USAGE.RENDER_ATTACHMENT | GPU_TEXTURE_USAGE.TEXTURE_BINDING,
      });
      this.sceneColorView = this.sceneColorTex.createView();

      if (aoW > 0 && aoH > 0) {
        this.aoTex = dev.createTexture({
          size: [aoW, aoH],
          format: 'rgba8unorm',
          usage: GPU_TEXTURE_USAGE.STORAGE_BINDING | GPU_TEXTURE_USAGE.TEXTURE_BINDING,
        });
        this.aoSampleView = this.aoTex.createView();
        this.aoStorageView = this.aoTex.createView();
        this.ssaoBg = dev.createBindGroup({
          layout: samples > 1 ? this.ssaoMsaaBgl : this.ssaoSingleBgl,
          entries: [
            { binding: 0, resource: { buffer: this.ssaoUniformBuf } },
            { binding: 1, resource: this.depthView },
            { binding: 2, resource: this.aoStorageView },
          ],
        });
      }

      const bw = Math.max(1, w >> 2), bh = Math.max(1, h >> 2);
      this.bloomTexA = dev.createTexture({
        size: [bw, bh],
        format: 'rgba16float',
        usage: GPU_TEXTURE_USAGE.STORAGE_BINDING | GPU_TEXTURE_USAGE.TEXTURE_BINDING,
      });
      this.bloomViewA = this.bloomTexA.createView();
      this.bloomTexB = dev.createTexture({
        size: [bw, bh],
        format: 'rgba16float',
        usage: GPU_TEXTURE_USAGE.STORAGE_BINDING | GPU_TEXTURE_USAGE.TEXTURE_BINDING,
      });
      this.bloomViewB = this.bloomTexB.createView();

      this.bloomExtBg = dev.createBindGroup({
        layout: this.bloomBgl,
        entries: [
          { binding: 0, resource: { buffer: this.bloomExtUniformBuf, offset: 0, size: 16 } },
          { binding: 1, resource: this.sceneColorView },
          { binding: 2, resource: this.sampClampLin },
          { binding: 3, resource: this.bloomViewA },
        ],
      });
      this.bloomBlurHBg = dev.createBindGroup({
        layout: this.bloomBgl,
        entries: [
          { binding: 0, resource: { buffer: this.bloomBlurUniformBuf, offset: 0, size: 16 } },
          { binding: 1, resource: this.bloomViewA },
          { binding: 2, resource: this.sampClampLin },
          { binding: 3, resource: this.bloomViewB },
        ],
      });
      this.bloomBlurVBg = dev.createBindGroup({
        layout: this.bloomBgl,
        entries: [
          { binding: 0, resource: { buffer: this.bloomBlurUniformBuf, offset: 0, size: 16 } },
          { binding: 1, resource: this.bloomViewB },
          { binding: 2, resource: this.sampClampLin },
          { binding: 3, resource: this.bloomViewA },
        ],
      });
      this.compBg = dev.createBindGroup({
        layout: this.compBgl,
        entries: [
          { binding: 0, resource: { buffer: this.compUniformBuf } },
          { binding: 1, resource: this.sceneColorView },
          { binding: 2, resource: this.bloomViewA },
          { binding: 3, resource: this.aoSampleView ?? this.whiteView },
          { binding: 4, resource: this.sampClampLin },
        ],
      });
    }

    this.tgtW = w;
    this.tgtH = h;
    this.tgtHdr = usePost;
    this.tgtMsaa = samples;
    this.tgtAoW = aoW;
    this.tgtAoH = aoH;
  }

  // =========================================================================
  // Persistent Scene Flatten & Cascade Computation
  // =========================================================================

  private rebuildSceneLists(scene: Scene): void {
    this.flatMeshes.length = 0;
    this.flatPoints.length = 0;
    this.flatLines.length = 0;
    this.flatSky = null;
    this.sunLight = null;
    this.hemiLight = null;
    this.spotLight = null;
    this.pointLight = null;

    scene.traverse((o) => {
      if (o instanceof DirectionalLight) {
        if (!this.sunLight) this.sunLight = o;
      } else if (o instanceof HemisphereLight) {
        if (!this.hemiLight) this.hemiLight = o;
      } else if (o instanceof SpotLight) {
        if (!this.spotLight) this.spotLight = o;
      } else if (o instanceof PointLight) {
        if (!this.pointLight) this.pointLight = o;
      } else if (o instanceof Points) {
        this.flatPoints.push(o);
      } else if (o instanceof LineSegments) {
        this.flatLines.push(o);
      } else if (o instanceof Mesh) {
        const m = o.material;
        if (!Array.isArray(m) && m.isShaderMaterial && o.renderOrder < 0) {
          this.flatSky = o;
        } else {
          this.flatMeshes.push(o);
        }
      }
    });

    this.cachedSceneId = scene.id;
    this.cachedStructVer = scene._structureVersion;
  }

  private allocDrawItem(
    obj: Mesh, geo: BufferGeometry, mat: Material,
    start: number, count: number, mode: DrawMode, modeId: number, depth: number, drawIdx: number,
    sx: number, sy: number, sz: number, sr: number,
  ): DrawItem {
    let it = this.drawPool[this.poolIdx];
    const sortKey = (obj.renderOrder + 100) * 1e9 + modeId * 1e7 + mat.id * 1e3 + (geo.id & 1023);
    if (!it) {
      it = { obj, geo, mat, start, count, mode, modeId, sortKey, depth, drawIdx, sx, sy, sz, sr };
      this.drawPool.push(it);
    } else {
      it.obj = obj; it.geo = geo; it.mat = mat; it.start = start; it.count = count;
      it.mode = mode; it.modeId = modeId; it.sortKey = sortKey; it.depth = depth; it.drawIdx = drawIdx;
      it.sx = sx; it.sy = sy; it.sz = sz; it.sr = sr;
    }
    this.poolIdx++;
    return it;
  }

  private buildLightViewMatrix(out: Float32Array, ex: number, ey: number, ez: number): void {
    const rx = _vRight.x, ry = _vRight.y, rz = _vRight.z;
    const ux = _vLightUp.x, uy = _vLightUp.y, uz = _vLightUp.z;
    const zx = _vSun.x, zy = _vSun.y, zz = _vSun.z;
    out[0] = rx; out[4] = ry; out[8] = rz; out[12] = -(rx * ex + ry * ey + rz * ez);
    out[1] = ux; out[5] = uy; out[9] = uz; out[13] = -(ux * ex + uy * ey + uz * ez);
    out[2] = zx; out[6] = zy; out[10] = zz; out[14] = -(zx * ex + zy * ey + zz * ez);
    out[3] = 0;  out[7] = 0;  out[11] = 0;  out[15] = 1;
  }

  private computeShadowCascades(camera: Camera): number {
    const sun = this.sunLight;
    if (!this.shadowMap.enabled || !this.shadowConfig.enabled || !sun || sun.intensity < 0.01) return 0;

    const sm = sun.matrixWorld.elements, tm = sun.target.matrixWorld.elements;
    _vSun.set(sm[12] - tm[12], sm[13] - tm[13], sm[14] - tm[14]).normalize();
    if (_vSun.lengthSq() < 1e-4) _vSun.set(0.3, 0.9, 0.2).normalize();
    const up = Math.abs(_vSun.y) > 0.98 ? _vAltUp : _vUp;
    _vRight.crossVectors(up, _vSun).normalize();
    _vLightUp.crossVectors(_vSun, _vRight).normalize();

    const splits = this.shadowConfig.cascades;
    const mapSize = this.shadowConfig.mapSize;

    if (!splits || splits.length === 0) {
      const sc = sun.shadow.camera;
      const half = Math.max(Math.abs(sc.right), Math.abs(sc.top), 75);
      const texel = (half * 2) / mapSize;
      const tx = tm[12], ty = tm[13], tz = tm[14];
      this.shadowFocusX = tx;
      this.shadowFocusZ = tz;
      this.shadowLowRange2 = (half * 1.45) ** 2;

      const worldLx = _vRight.x * tx + _vRight.y * ty + _vRight.z * tz;
      const worldLy = _vLightUp.x * tx + _vLightUp.y * ty + _vLightUp.z * tz;
      const dx = Math.round(worldLx / texel) * texel - worldLx;
      const dy = Math.round(worldLy / texel) * texel - worldLy;

      const ex = tx + _vSun.x * 240, ey = ty + _vSun.y * 240, ez = tz + _vSun.z * 240;
      const lightNear = 1, lightFar = 560;
      this.buildLightViewMatrix(_lightView, ex, ey, ez);
      _tmpMat4.makeOrthographic(dx - half, dx + half, dy + half, dy - half, lightNear, lightFar);
      _lightProj.set(_tmpMat4.elements);
      multiplyMat4Arrays(_lightVpGl, _lightProj, _lightView);
      extractFrustumPlanes(_lightVpGl, _shPlanesAll[0]);
      toZeroOneClipZ(this.shadowMatsZo, _lightVpGl, 0);
      this.cascadeSplits[0] = half * 1.15;
      this.cascadeBiases[0] = Math.max(0.04, texel * 1.2);
      this.cascadeDepthBiases[0] = (0.03 + texel * 1.5) / (lightFar - lightNear);
      return 1;
    }

    const nCascades = Math.min(splits.length, MAX_CASCADES);
    const pCam = camera as PerspectiveCamera;
    const fovRad = ((pCam.fov || 62) * Math.PI) / 180;
    const tanH = Math.tan(fovRad * 0.5);
    const aspect = pCam.aspect || (this.cssW / Math.max(1, this.cssH));
    const camM = camera.matrixWorld.elements;
    let prevSplit = camera.near;

    for (let c = 0; c < nCascades; c++) {
      const farSplit = splits[c];
      const nearSplit = prevSplit;
      prevSplit = farSplit;
      this.cascadeSplits[c] = farSplit;

      const nh = nearSplit * tanH, nw = nh * aspect;
      const fh = farSplit * tanH, fw = fh * aspect;
      _cx[0] = -nw; _cx[1] = nw; _cx[2] = -nw; _cx[3] = nw; _cx[4] = -fw; _cx[5] = fw; _cx[6] = -fw; _cx[7] = fw;
      _cy[0] = -nh; _cy[1] = -nh; _cy[2] = nh; _cy[3] = nh; _cy[4] = -fh; _cy[5] = -fh; _cy[6] = fh; _cy[7] = fh;
      _cz[0] = -nearSplit; _cz[1] = -nearSplit; _cz[2] = -nearSplit; _cz[3] = -nearSplit;
      _cz[4] = -farSplit;  _cz[5] = -farSplit;  _cz[6] = -farSplit;  _cz[7] = -farSplit;

      let cx = 0, cy = 0, cz = 0;
      for (let i = 0; i < 8; i++) {
        const lx = _cx[i], ly = _cy[i], lz = _cz[i];
        const x = camM[0] * lx + camM[4] * ly + camM[8] * lz + camM[12];
        const y = camM[1] * lx + camM[5] * ly + camM[9] * lz + camM[13];
        const z = camM[2] * lx + camM[6] * ly + camM[10] * lz + camM[14];
        _wx[i] = x; _wy[i] = y; _wz[i] = z;
        cx += x; cy += y; cz += z;
      }
      cx *= 0.125; cy *= 0.125; cz *= 0.125;
      let radius = 0;
      for (let i = 0; i < 8; i++) {
        const d = Math.hypot(_wx[i] - cx, _wy[i] - cy, _wz[i] - cz);
        if (d > radius) radius = d;
      }
      radius = Math.ceil(radius * 2) * 0.5;
      const texel = (radius * 2) / mapSize;

      const worldLx = _vRight.x * cx + _vRight.y * cy + _vRight.z * cz;
      const worldLy = _vLightUp.x * cx + _vLightUp.y * cy + _vLightUp.z * cz;
      const dx = Math.round(worldLx / texel) * texel - worldLx;
      const dy = Math.round(worldLy / texel) * texel - worldLy;

      const ex = cx + _vSun.x * (radius + 420);
      const ey = cy + _vSun.y * (radius + 420);
      const ez = cz + _vSun.z * (radius + 420);
      const lightNear = 1, lightFar = radius * 2 + 650;
      this.buildLightViewMatrix(_lightView, ex, ey, ez);

      _tmpMat4.makeOrthographic(dx - radius, dx + radius, dy + radius, dy - radius, lightNear, lightFar);
      _lightProj.set(_tmpMat4.elements);
      multiplyMat4Arrays(_lightVpGl, _lightProj, _lightView);
      extractFrustumPlanes(_lightVpGl, _shPlanesAll[c]);
      toZeroOneClipZ(this.shadowMatsZo, _lightVpGl, c * 16);
      this.cascadeBiases[c] = Math.max(0.02, texel * 1.2);
      this.cascadeDepthBiases[c] = (0.03 + texel * 1.5) / (lightFar - lightNear);
    }
    return nCascades;
  }

  private ensureDrawCapacity(needed: number): void {
    if (needed <= this.drawStorageCap) return;
    while (this.drawStorageCap < needed) this.drawStorageCap *= 2;
    this.drawStorageBuf.destroy();
    this.drawStorageBuf = this.device.createBuffer({
      size: this.drawStorageCap * DRAW_STRIDE_BYTES,
      usage: GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_DST,
    });
    const nextBuf = new ArrayBuffer(this.drawStorageCap * DRAW_STRIDE_BYTES);
    new Uint8Array(nextBuf).set(new Uint8Array(this.drawRawBuf));
    this.drawRawBuf = nextBuf;
    this.drawF32 = new Float32Array(this.drawRawBuf);
    this.drawU32 = new Uint32Array(this.drawRawBuf);
    this.rebuildDrawBoneBg();
  }

  private ensureBoneCapacity(neededMats: number): void {
    if (neededMats <= this.boneStorageCap) return;
    while (this.boneStorageCap < neededMats) this.boneStorageCap *= 2;
    this.boneStorageBuf.destroy();
    this.boneStorageBuf = this.device.createBuffer({
      size: this.boneStorageCap * 64,
      usage: GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_DST,
    });
    const nextCpu = new Float32Array(this.boneStorageCap * 16);
    nextCpu.set(this.boneCpu);
    this.boneCpu = nextCpu;
    this.rebuildDrawBoneBg();
  }

  private ensureCullCapacity(neededJobs: number): void {
    if (neededJobs <= this.cullParamsCap) return;
    while (this.cullParamsCap < neededJobs) this.cullParamsCap *= 2;
    this.cullParamsBuf.destroy();
    this.cullParamsBuf = this.device.createBuffer({
      size: this.cullParamsCap * this.uAlign,
      usage: GPU_BUFFER_USAGE.UNIFORM | GPU_BUFFER_USAGE.COPY_DST,
    });
    this.cullParamsCpu = new ArrayBuffer(this.cullParamsCap * this.uAlign);
    this.cullParamsF32 = new Float32Array(this.cullParamsCpu);
    this.cullParamsU32 = new Uint32Array(this.cullParamsCpu);
    for (const e of this.instCache.values()) {
      e.cullBg = this.device.createBindGroup({
        layout: this.cullBgl,
        entries: [
          { binding: 0, resource: { buffer: this.cullParamsBuf, offset: 0, size: 112 } },
          { binding: 1, resource: { buffer: e.sphereBuf } },
          { binding: 2, resource: { buffer: e.visBuf } },
          { binding: 3, resource: { buffer: e.indirectBuf } },
        ],
      });
    }
  }

  private writeDrawData(
    drawIdx: number,
    mesh: Mesh,
    mat: Material,
    hasInstColor: boolean,
    toneMapInline: boolean,
    boneOffset: number,
  ): void {
    this.ensureDrawCapacity(drawIdx + 1);
    const f = this.drawF32;
    const u = this.drawU32;
    const o = drawIdx * DRAW_STRIDE_FLOATS;
    const mw = mesh.matrixWorld.elements;
    for (let k = 0; k < 16; k++) f[o + k] = mw[k];

    f[o + 16] = mat.color.r;
    f[o + 17] = mat.color.g;
    f[o + 18] = mat.color.b;
    f[o + 19] = mat.opacity;

    f[o + 20] = mat.emissive.r * mat.emissiveIntensity;
    f[o + 21] = mat.emissive.g * mat.emissiveIntensity;
    f[o + 22] = mat.emissive.b * mat.emissiveIntensity;
    f[o + 23] = mat.alphaTest;

    f[o + 24] = mat.roughness;
    f[o + 25] = mat.metalness;
    f[o + 26] = mat.clearcoat;
    f[o + 27] = mat.clearcoatRoughness;

    f[o + 28] = mat.bumpScale;
    f[o + 29] = mat.aoMapIntensity;
    f[o + 30] = mat.envMapIntensity;
    f[o + 31] = 0;

    if (mat.map) {
      f[o + 32] = mat.map.repeat.x; f[o + 33] = mat.map.repeat.y;
      f[o + 34] = mat.map.offset.x; f[o + 35] = mat.map.offset.y;
    } else {
      f[o + 32] = 1; f[o + 33] = 1; f[o + 34] = 0; f[o + 35] = 0;
    }

    if (mat.personOutfit) {
      const po = mat.personOutfit;
      f[o + 36] = po.top.x; f[o + 37] = po.top.y; f[o + 38] = po.top.z; f[o + 39] = po.top.w;
      f[o + 40] = po.bottom.x; f[o + 41] = po.bottom.y; f[o + 42] = po.bottom.z; f[o + 43] = po.bottom.w;
      f[o + 44] = po.ref.x; f[o + 45] = po.ref.y;
    } else {
      f[o + 36] = 0; f[o + 37] = 0; f[o + 38] = 0; f[o + 39] = 0;
      f[o + 40] = 0; f[o + 41] = 0; f[o + 42] = 0; f[o + 43] = 0;
      f[o + 44] = 1; f[o + 45] = 1;
    }

    let flags = 0;
    if (mat.vertexColors || hasInstColor) flags |= FLAG_VERTEX_COLORS;
    if (mat.carPaintShell) flags |= FLAG_CAR_SHELL;
    if (mat.personOutfit) flags |= FLAG_PERSON_OUTFIT;
    if (mat.roughnessMap || mat.metalnessMap || mat.clearcoatMap) flags |= FLAG_CRM_MAP;
    if (mat.bumpMap) flags |= FLAG_BUMP_MAP;
    if (mat.aoMap) flags |= FLAG_AO_MAP;
    if ((mat.bumpMap && mat.bumpMap.channel === 1) || (mat.aoMap && mat.aoMap.channel === 1)) flags |= FLAG_BUMP_AO_UV1;
    if (mat.flatShading) flags |= FLAG_FLAT_SHADING;
    if (mat.fog) flags |= FLAG_FOG;
    if (mesh.receiveShadow) flags |= FLAG_RECEIVE_SHADOW;
    if (mat.isMeshBasicMaterial) flags |= FLAG_UNLIT;
    if (toneMapInline) flags |= FLAG_TONEMAP_INLINE;
    if (mat.emissiveMap) flags |= FLAG_EMISSIVE_MAP;
    if (mat.alphaToCoverage && this.msaaSamples > 1) flags |= FLAG_ALPHA_TO_COV;

    u[o + 46] = flags >>> 0;
    u[o + 47] = boneOffset >>> 0;
  }

  // =========================================================================
  // Main Frame Render (GPU-Driven Compute Culling + Shadow + MSAA + Compute Post)
  // =========================================================================

  render(scene: Scene, camera: Camera): void {
    if (this.deviceLost || this.fbW <= 0 || this.fbH <= 0) return;

    const dev = this.device;
    const queue = dev.queue;
    this.info.render.calls = 0;
    this.info.render.triangles = 0;

    scene.updateMatrixWorld(false);
    camera.updateMatrixWorld(false);

    if (this.cachedSceneId !== scene.id || this.cachedStructVer !== scene._structureVersion) {
      this.rebuildSceneLists(scene);
    }

    multiplyMat4Arrays(_vpGl, camera.projectionMatrix.elements, camera.matrixWorldInverse.elements);
    extractFrustumPlanes(_vpGl, _camPlanes);
    toZeroOneClipZ(_vpZo, _vpGl);

    const camM = camera.matrixWorld.elements;
    const camX = camM[12], camY = camM[13], camZ = camM[14];

    if (this.flatSky && this.flatSky.material instanceof ShaderMaterial) {
      const u = this.flatSky.material.uniforms;
      const top = u.top?.value as { r: number; g: number; b: number } | undefined;
      const hor = u.horizon?.value as { r: number; g: number; b: number } | undefined;
      if (top) { this.skyTop[0] = top.r; this.skyTop[1] = top.g; this.skyTop[2] = top.b; }
      if (hor) { this.skyHor[0] = hor.r; this.skyHor[1] = hor.g; this.skyHor[2] = hor.b; }
    }

    const usePost = this.postConfig.enabled;
    const toneMapInline = !usePost;
    const cCount = this.computeShadowCascades(camera);
    const moverFar2 = cCount > 1 ? (this.cascadeSplits[cCount - 2] * 1.15) ** 2 : this.shadowLowRange2;

    this.ensureRenderTargets(this.fbW, this.fbH, usePost, this.postConfig.aoScale);
    if (cCount > 0) {
      this.ensureShadowTargets(this.shadowConfig.mapSize, cCount);
      this.rebuildSceneBg(this.shadowArrayView!);
    } else {
      this.rebuildSceneBg(this.dummyShadowView);
    }

    // 1. Collect draw queues, pack single DrawData & Bone Storage buffers, and schedule GPU Compute Cull jobs!
    this.opaqueQueue.length = 0;
    this.transparentQueue.length = 0;
    this.shadowQueue.length = 0;
    this.poolIdx = 0;
    this.cullJobs.length = 0;
    const activeInsts = this.activeInstsSet;
    activeInsts.clear();

    let drawCount = 0;
    let boneMatOffset = 0;
    const meshes = this.flatMeshes;

    for (let i = 0; i < meshes.length; i++) {
      const mesh = meshes[i];
      if (!isObjVisible(mesh)) continue;
      const geo = mesh.geometry;
      if (!geo.attributes.position || geo.attributes.position.count === 0) continue;

      const isInst = (mesh as InstancedMesh).isInstancedMesh === true;
      if (isInst && (mesh as InstancedMesh).count <= 0) continue;
      const isSkin = (mesh as SkinnedMesh).isSkinnedMesh === true
        && Boolean(geo.attributes.skinIndex && geo.attributes.skinWeight);
      const mode: DrawMode = isInst ? 'instanced' : isSkin ? 'skinned' : 'static';
      const modeId = isInst ? 1 : isSkin ? 2 : 0;

      const mw = mesh.matrixWorld.elements;
      let sx = mw[12], sy = mw[13], sz = mw[14], sr = 1e9;
      let inMainFrustum = true;
      if (mesh.frustumCulled) {
        const bs = mesh.boundingSphere ?? geo.boundingSphere ?? geo.computeBoundingSphere();
        sx = mw[0] * bs.center.x + mw[4] * bs.center.y + mw[8] * bs.center.z + mw[12];
        sy = mw[1] * bs.center.x + mw[5] * bs.center.y + mw[9] * bs.center.z + mw[13];
        sz = mw[2] * bs.center.x + mw[6] * bs.center.y + mw[10] * bs.center.z + mw[14];
        const scX = mw[0] * mw[0] + mw[1] * mw[1] + mw[2] * mw[2];
        const scY = mw[4] * mw[4] + mw[5] * mw[5] + mw[6] * mw[6];
        const scZ = mw[8] * mw[8] + mw[9] * mw[9] + mw[10] * mw[10];
        sr = bs.radius * Math.sqrt(Math.max(scX, scY, scZ));
        inMainFrustum = sphereIntersectsFrustum(_camPlanes, sx, sy, sz, sr);
      }

      const dist2 = (sx - camX) ** 2 + (sy - camY) ** 2 + (sz - camZ) ** 2;
      const inShadowRange = !isSkin || (cCount === 1
        ? (sx - this.shadowFocusX) ** 2 + (sz - this.shadowFocusZ) ** 2 < this.shadowLowRange2
        : dist2 < moverFar2 * 1.4);
      const castSh = cCount > 0 && mesh.castShadow && inShadowRange;
      if (!inMainFrustum && !castSh) continue;

      let myBoneOffset = 0;
      if (isSkin) {
        const sm = mesh as SkinnedMesh;
        if (sm.skeleton) {
          invertMat4Array(_meshWorldInv, mw);
          if (sm._hasNonIdentityBind) {
            sm.skeleton.update(_meshWorldInv, sm.bindMatrix.elements, sm.bindMatrixInverse.elements);
          } else {
            sm.skeleton.update(_meshWorldInv);
          }
          const nb = Math.min(sm.skeleton.bones.length, MAX_BONES);
          this.ensureBoneCapacity(boneMatOffset + nb);
          this.boneCpu.set(sm.skeleton.boneMatrices.subarray(0, nb * 16), boneMatOffset * 16);
          myBoneOffset = boneMatOffset;
          boneMatOffset += nb;
        }
      }

      const mats = Array.isArray(mesh.material) ? mesh.material : null;
      const groups = geo.groups;
      const hasInstColor = isInst && (mesh as InstancedMesh).instanceColor !== null;

      const emitGroup = (mat: Material, start: number, count: number) => {
        if (!mat || !mat.visible || count <= 0) return;
        const myDrawIdx = drawCount++;
        this.writeDrawData(myDrawIdx, mesh, mat, hasInstColor, toneMapInline, myBoneOffset);
        this.getMaterialBindGroup(mat);

        if (isInst) {
          const geoRec = this.syncGeometry(geo);
          const instGpu = this.syncInstanced(mesh as InstancedMesh, geoRec);
          if (!activeInsts.has(instGpu)) {
            activeInsts.add(instGpu);
            const uStride = this.uAlign / 4;
            const sc = instGpu.slotCpu;
            // Slot 0 = main pass, slots 1..cCount = shadow cascades
            for (let s = 0; s <= cCount; s++) {
              const off = s * uStride;
              sc[off] = myDrawIdx;
              sc[off + 1] = s * instGpu.total;
              sc[off + 2] = hasInstColor ? 1 : 0;
              sc[off + 3] = 0;
            }
            queue.writeBuffer(instGpu.slotUniformBuf, 0, sc, 0, (cCount + 1) * uStride);
            if (inMainFrustum) {
              this.cullJobs.push({ inst: instGpu, slot: 0, planes: _camPlanes });
            }
            if (castSh && !mat.transparent) {
              for (let c = 0; c < cCount; c++) {
                this.cullJobs.push({ inst: instGpu, slot: 1 + c, planes: _shPlanesAll[c] });
              }
            }
          }
        }

        if (inMainFrustum) {
          const item = this.allocDrawItem(mesh, geo, mat, start, count, mode, modeId, dist2, myDrawIdx, sx, sy, sz, sr);
          if (mat.transparent) this.transparentQueue.push(item);
          else this.opaqueQueue.push(item);
        }
        if (castSh && !mat.transparent) {
          this.shadowQueue.push(this.allocDrawItem(mesh, geo, mat, start, count, mode, modeId, dist2, myDrawIdx, sx, sy, sz, sr));
        }
      };

      if (mats && groups.length > 0) {
        for (let g = 0; g < groups.length; g++) {
          const gr = groups[g];
          emitGroup(mats[gr.materialIndex], gr.start, gr.count);
        }
      } else {
        const mat = mats ? mats[0] : (mesh.material as Material);
        const total = geo.index ? geo.index.count : geo.attributes.position.count;
        emitGroup(mat, 0, total);
      }
    }

    this.opaqueQueue.sort(sortOpaque);
    this.transparentQueue.sort(sortTransparent);
    if (cCount > 0) this.shadowQueue.sort(sortOpaque);

    // 2. Upload ALL batched frame buffers ONCE before encoding commands!
    this.uploadSceneUniforms(scene, camera, cCount);
    if (drawCount > 0) {
      queue.writeBuffer(this.drawStorageBuf, 0, this.drawRawBuf, 0, drawCount * DRAW_STRIDE_BYTES);
    }
    if (boneMatOffset > 0) {
      queue.writeBuffer(this.boneStorageBuf, 0, this.boneCpu, 0, boneMatOffset * 16);
    }
    if (cCount > 0) {
      for (let c = 0; c < cCount; c++) {
        queue.writeBuffer(this.shadowPassUniformBuf, c * this.uAlign, this.shadowMatsZo, c * 16, 16);
      }
    }
    if (this.cullJobs.length > 0) {
      this.ensureCullCapacity(this.cullJobs.length);
      const stride = this.uAlign / 4;
      for (let j = 0; j < this.cullJobs.length; j++) {
        const job = this.cullJobs[j];
        const o = j * stride;
        const pl = job.planes;
        for (let k = 0; k < 24; k++) this.cullParamsF32[o + k] = pl[k];
        this.cullParamsU32[o + 24] = job.inst.total;
        this.cullParamsU32[o + 25] = job.slot;
        this.cullParamsU32[o + 26] = job.slot * job.inst.total;
        this.cullParamsU32[o + 27] = 0;
      }
      queue.writeBuffer(this.cullParamsBuf, 0, this.cullParamsCpu, 0, this.cullJobs.length * this.uAlign);
    }

    // Prepare dynamic Sky / Points / Lines uniform slots before encoding
    const hasSky = Boolean(this.flatSky && isObjVisible(this.flatSky));
    if (hasSky) this.uploadSkyUniforms(this.flatSky!, toneMapInline);
    const activeLines = this.prepareLines(toneMapInline);
    const activePoints = this.preparePoints(camera, toneMapInline);

    // 3. Begin single frame GPUCommandEncoder
    const enc = dev.createCommandEncoder();
    this.flushPendingMipmaps(enc);

    // 3a. GPU Compute Frustum Culling Pass for ALL InstancedMeshes across main + all cascades!
    if (this.cullJobs.length > 0) {
      for (const inst of activeInsts) {
        enc.copyBufferToBuffer(inst.indirectTemplate, 0, inst.indirectBuf, 0, 5 * 20);
      }
      const cpass = enc.beginComputePass();
      cpass.setPipeline(this.cullPipe);
      for (let j = 0; j < this.cullJobs.length; j++) {
        const job = this.cullJobs[j];
        cpass.setBindGroup(0, job.inst.cullBg, [j * this.uAlign]);
        cpass.dispatchWorkgroups(Math.ceil(job.inst.total / 64));
      }
      cpass.end();
    }

    // 3b. Shadow Map Array Passes (Hardware CSM + Indirect Instanced Draws)
    if (cCount > 0) {
      this.encodeShadowPasses(enc, cCount, moverFar2);
    }

    // 3c. Main Scene Pass (Direct 4x MSAA with single-pass hardware resolveTarget!)
    const swapTex = this.ctx.getCurrentTexture();
    const swapView = swapTex.createView();
    const resolveTarget = usePost ? this.sceneColorView! : swapView;
    const colorAttachment = this.msaaSamples > 1
      ? {
        view: this.msaaColorView!,
        resolveTarget,
        loadOp: 'clear',
        storeOp: 'discard', // Tile memory discard after MSAA resolve — huge TBDR bandwidth win!
        clearValue: { r: 0.5, g: 0.7, b: 0.9, a: 1.0 },
      }
      : {
        view: resolveTarget,
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0.5, g: 0.7, b: 0.9, a: 1.0 },
      };

    const mainPass = enc.beginRenderPass({
      colorAttachments: [colorAttachment],
      depthStencilAttachment: {
        view: this.depthView!,
        depthLoadOp: 'clear',
        depthStoreOp: (usePost && this.tgtAoW > 0) ? 'store' : 'discard',
        depthClearValue: 1.0,
      },
    });

    if (hasSky) {
      this.encodeSky(mainPass, this.flatSky!, usePost);
    }

    mainPass.setBindGroup(0, this.sceneBg);
    mainPass.setBindGroup(1, this.drawBoneBg);

    this.encodeMeshQueue(mainPass, this.opaqueQueue, usePost);
    this.encodeMeshQueue(mainPass, this.transparentQueue, usePost);
    if (activeLines > 0) this.encodeLines(mainPass, usePost);
    if (activePoints > 0) this.encodePoints(mainPass, usePost);
    mainPass.end();

    // 3d. Compute Post-Processing (Zero-Resolve Multisampled SSAO + Compute Bloom + ACES Composite!)
    if (usePost) {
      this.encodePostProcess(enc, camera, swapView);
    }

    queue.submit([enc.finish()]);
  }

  private uploadSceneUniforms(scene: Scene, camera: Camera, cCount: number): void {
    const f = this.sceneUniformCpu;
    const camM = camera.matrixWorld.elements;
    const sun = this.sunLight;
    const hemi = this.hemiLight;
    const spot = this.spotLight;
    const point = this.pointLight;
    const fog = scene.fog;

    f.set(_vpZo, 0);
    f.set(camera.matrixWorldInverse.elements, 16);

    const envBase = this.postConfig.reflections ? scene.environmentIntensity * 1.25 : 0;
    f[32] = camM[12]; f[33] = camM[13]; f[34] = camM[14]; f[35] = this.toneMappingExposure;

    if (sun) {
      const sm = sun.matrixWorld.elements, tm = sun.target.matrixWorld.elements;
      _vSun.set(sm[12] - tm[12], sm[13] - tm[13], sm[14] - tm[14]).normalize();
      f[40] = sun.color.r * sun.intensity;
      f[41] = sun.color.g * sun.intensity;
      f[42] = sun.color.b * sun.intensity;
    } else {
      _vSun.set(0.3, 0.9, 0.2).normalize();
      f[40] = 2.5; f[41] = 2.4; f[42] = 2.2;
    }
    f[36] = _vSun.x; f[37] = _vSun.y; f[38] = _vSun.z; f[39] = envBase;
    f[43] = cCount;

    if (hemi) {
      f[44] = hemi.color.r * hemi.intensity; f[45] = hemi.color.g * hemi.intensity; f[46] = hemi.color.b * hemi.intensity;
      f[48] = hemi.groundColor.r * hemi.intensity; f[49] = hemi.groundColor.g * hemi.intensity; f[50] = hemi.groundColor.b * hemi.intensity;
    } else {
      f[44] = 0.8; f[45] = 0.9; f[46] = 1.0;
      f[48] = 0.5; f[49] = 0.48; f[50] = 0.45;
    }
    f[47] = fog ? fog.far : 2e6;
    f[51] = this.pixelRatio;

    f[52] = this.skyTop[0]; f[53] = this.skyTop[1]; f[54] = this.skyTop[2]; f[55] = this.fbW;
    f[56] = this.skyHor[0]; f[57] = this.skyHor[1]; f[58] = this.skyHor[2]; f[59] = this.fbH;

    if (fog) {
      f[60] = fog.color.r; f[61] = fog.color.g; f[62] = fog.color.b; f[63] = fog.near;
    } else {
      f[60] = 0.7; f[61] = 0.8; f[62] = 0.9; f[63] = 1e6;
    }

    let spDecay = 2;
    if (spot && spot.visible && spot.intensity > 0.01) {
      const sm = spot.matrixWorld.elements, tm = spot.target.matrixWorld.elements;
      const sx = sm[12], sy = sm[13], sz = sm[14];
      _vSpotDir.set(tm[12] - sx, tm[13] - sy, tm[14] - sz).normalize();
      f[64] = sx; f[65] = sy; f[66] = sz; f[67] = spot.distance > 0 ? spot.distance : 70;
      f[68] = _vSpotDir.x; f[69] = _vSpotDir.y; f[70] = _vSpotDir.z; f[71] = Math.cos(spot.angle);
      f[72] = spot.color.r * spot.intensity; f[73] = spot.color.g * spot.intensity; f[74] = spot.color.b * spot.intensity;
      f[75] = Math.cos(spot.angle * (1 - spot.penumbra));
      spDecay = spot.decay ?? 2;
    } else {
      f.fill(0, 64, 76);
      f[71] = 1; f[75] = 1;
    }

    if (point && point.visible && point.intensity > 0.01) {
      const pm = point.matrixWorld.elements;
      f[76] = pm[12]; f[77] = pm[13]; f[78] = pm[14]; f[79] = point.distance > 0 ? point.distance : 60;
      f[80] = point.color.r * point.intensity; f[81] = point.color.g * point.intensity; f[82] = point.color.b * point.intensity;
      f[83] = point.decay ?? 1.6;
    } else {
      f.fill(0, 76, 84);
      f[83] = 1.6;
    }

    f[84] = spDecay;
    f[85] = this.shadowTexSize || 1024;
    f[86] = 0; f[87] = 0;

    f.set(this.cascadeSplits, 88);
    f.set(this.cascadeBiases, 92);
    f.set(this.cascadeDepthBiases, 96);
    f.set(this.shadowMatsZo, 100);

    this.device.queue.writeBuffer(this.sceneUniformBuf, 0, f);
  }

  private encodeShadowPasses(enc: GPUCommandEncoderLike, cCount: number, moverFar2: number): void {
    const queue = this.shadowQueue;
    for (let c = 0; c < cCount; c++) {
      const pass = enc.beginRenderPass({
        colorAttachments: [],
        depthStencilAttachment: {
          view: this.shadowLayerViews[c],
          depthLoadOp: 'clear',
          depthStoreOp: 'store',
          depthClearValue: 1.0,
        },
      });
      pass.setBindGroup(0, this.shadowPassBg, [c * this.uAlign]);
      pass.setBindGroup(1, this.drawBoneBg);

      const planes = _shPlanesAll[c];
      const isFarthest = cCount > 1 && c === cCount - 1;
      let lastPipe: GPURenderPipelineLike | null = null;
      let lastMatId = -1;

      for (let i = 0; i < queue.length; i++) {
        const it = queue[i];
        if (isFarthest && it.mode === 'skinned' && it.depth >= moverFar2) continue;
        if (it.obj.frustumCulled && !sphereIntersectsFrustum(planes, it.sx, it.sy, it.sz, it.sr)) continue;

        const pipe = this.getShadowPipeline(it.mode, it.modeId, it.mat.side === DoubleSide ? DoubleSide : FrontSide);
        if (lastPipe !== pipe) {
          pass.setPipeline(pipe);
          lastPipe = pipe;
        }
        if (lastMatId !== it.mat.id) {
          pass.setBindGroup(2, this.getMaterialBindGroup(it.mat));
          lastMatId = it.mat.id;
        }

        const geoRec = this.syncGeometry(it.geo);
        pass.setVertexBuffer(0, geoRec.vbuf);
        if (it.mode === 'skinned' && geoRec.skinBuf) pass.setVertexBuffer(1, geoRec.skinBuf);
        if (geoRec.ibuf) pass.setIndexBuffer(geoRec.ibuf, geoRec.indexFormat);

        if (it.mode === 'instanced') {
          const instGpu = this.instCache.get(it.obj.id)!;
          pass.setBindGroup(3, instGpu.renderBg, [(1 + c) * this.uAlign]);
          if (instGpu.isIndexed) pass.drawIndexedIndirect(instGpu.indirectBuf, (1 + c) * 20);
          else pass.drawIndirect(instGpu.indirectBuf, (1 + c) * 20);
          this.info.render.calls++;
          this.info.render.triangles += ((it.count / 3) | 0) * ((instGpu.total * 0.4) | 0);
        } else {
          if (geoRec.ibuf) pass.drawIndexed(it.count, 1, it.start, 0, it.drawIdx);
          else pass.draw(it.count, 1, it.start, it.drawIdx);
          this.info.render.calls++;
          this.info.render.triangles += (it.count / 3) | 0;
        }
      }
      pass.end();
    }
  }

  private encodeMeshQueue(pass: GPURenderPassEncoderLike, queue: DrawItem[], isHdr: boolean): void {
    const samples = this.msaaSamples;
    let lastPipe: GPURenderPipelineLike | null = null;
    let lastMatId = -1;

    for (let i = 0; i < queue.length; i++) {
      const it = queue[i];
      const mat = it.mat;
      const blendMode = mat.transparent ? (mat.blending === AdditiveBlending ? 2 : 1) : 0;
      const polyKey = mat.polygonOffset ? 1 : 0;
      const pipe = this.getPbrPipeline(it.mode, it.modeId, mat.side, blendMode, mat.depthWrite, polyKey, mat.alphaToCoverage, samples, isHdr);
      if (lastPipe !== pipe) {
        pass.setPipeline(pipe);
        lastPipe = pipe;
      }
      if (lastMatId !== mat.id) {
        pass.setBindGroup(2, this.getMaterialBindGroup(mat));
        lastMatId = mat.id;
      }

      const geoRec = this.syncGeometry(it.geo);
      pass.setVertexBuffer(0, geoRec.vbuf);
      if (it.mode === 'skinned' && geoRec.skinBuf) pass.setVertexBuffer(1, geoRec.skinBuf);
      if (geoRec.ibuf) pass.setIndexBuffer(geoRec.ibuf, geoRec.indexFormat);

      if (it.mode === 'instanced') {
        const instGpu = this.instCache.get(it.obj.id)!;
        pass.setBindGroup(3, instGpu.renderBg, [0]);
        if (instGpu.isIndexed) pass.drawIndexedIndirect(instGpu.indirectBuf, 0);
        else pass.drawIndirect(instGpu.indirectBuf, 0);
        this.info.render.calls++;
        this.info.render.triangles += ((it.count / 3) | 0) * ((instGpu.total * 0.4) | 0);
      } else {
        if (geoRec.ibuf) pass.drawIndexed(it.count, 1, it.start, 0, it.drawIdx);
        else pass.draw(it.count, 1, it.start, it.drawIdx);
        this.info.render.calls++;
        this.info.render.triangles += (it.count / 3) | 0;
      }
    }
  }

  private uploadSkyUniforms(sky: Mesh, toneMapInline: boolean): void {
    const f = this.skyUniformCpu;
    const mat = sky.material as ShaderMaterial;
    const u = mat.uniforms;
    f.set(_vpZo, 0);
    f.set(sky.matrixWorld.elements, 16);
    const getC = (name: string) => (u[name]?.value as { r: number; g: number; b: number } | undefined);
    const getV = (name: string) => (u[name]?.value as { x: number; y: number; z: number } | undefined);
    const getF = (name: string) => (u[name]?.value as number | undefined) ?? 0;

    const top = getC('top');
    const hor = getC('horizon');
    const sd = getV('sunDir');
    const sc = getC('sunColor');
    const cc = getC('cloudColor');

    f[32] = top?.r ?? 0.2; f[33] = top?.g ?? 0.45; f[34] = top?.b ?? 0.8; f[35] = getF('stars');
    f[36] = hor?.r ?? 0.7; f[37] = hor?.g ?? 0.85; f[38] = hor?.b ?? 0.94; f[39] = getF('time');
    f[40] = sd?.x ?? 0.3;  f[41] = sd?.y ?? 0.9;   f[42] = sd?.z ?? 0.2;  f[43] = getF('cover');
    f[44] = sc?.r ?? 1.0;  f[45] = sc?.g ?? 0.95;  f[46] = sc?.b ?? 0.85; f[47] = getF('moon');
    f[48] = cc?.r ?? 1.0;  f[49] = cc?.g ?? 1.0;   f[50] = cc?.b ?? 1.0;  f[51] = this.toneMappingExposure;
    f[52] = toneMapInline ? 1 : 0; f[53] = 0; f[54] = 0; f[55] = 0;

    this.device.queue.writeBuffer(this.skyUniformBuf, 0, f);
  }

  private encodeSky(pass: GPURenderPassEncoderLike, sky: Mesh, isHdr: boolean): void {
    const geoRec = this.syncGeometry(sky.geometry);
    pass.setPipeline(this.getSkyPipeline(this.msaaSamples, isHdr));
    pass.setBindGroup(0, this.skyBg);
    pass.setVertexBuffer(0, geoRec.vbuf);
    if (geoRec.ibuf) {
      pass.setIndexBuffer(geoRec.ibuf, geoRec.indexFormat);
      pass.drawIndexed(geoRec.indexCount);
    } else {
      pass.draw(geoRec.vertexCount);
    }
    this.info.render.calls++;
    this.info.render.triangles += ((geoRec.indexCount || geoRec.vertexCount) / 3) | 0;
  }

  private prepareLines(toneMapInline: boolean): number {
    let slot = 0;
    const stride = this.uAlign / 4;
    for (let i = 0; i < this.flatLines.length && slot < MAX_LINES_SLOTS; i++) {
      const ls = this.flatLines[i];
      if (!isObjVisible(ls)) continue;
      const posAttr = ls.geometry.attributes.position;
      if (!posAttr || posAttr.count === 0) continue;

      let rec = this.linesCache.get(ls.id);
      if (!rec || rec.cap < posAttr.count) {
        if (rec) rec.buf.destroy();
        const cap = Math.max(posAttr.count, 64);
        rec = {
          buf: this.device.createBuffer({ size: cap * 12, usage: GPU_BUFFER_USAGE.VERTEX | GPU_BUFFER_USAGE.COPY_DST }),
          cap,
          posVer: -1,
          count: 0,
        };
        this.linesCache.set(ls.id, rec);
      }
      if (rec.posVer !== posAttr.version) {
        this.device.queue.writeBuffer(rec.buf, 0, posAttr.array, 0, posAttr.count * 3);
        rec.posVer = posAttr.version;
        rec.count = posAttr.count;
      }

      const o = slot * stride;
      const f = this.linesUniformCpu;
      f.set(_vpZo, o);
      f.set(ls.matrixWorld.elements, o + 16);
      f[o + 32] = ls.material.color.r; f[o + 33] = ls.material.color.g; f[o + 34] = ls.material.color.b; f[o + 35] = ls.material.opacity;
      f[o + 36] = toneMapInline ? 1 : 0; f[o + 37] = this.toneMappingExposure; f[o + 38] = 0; f[o + 39] = 0;
      slot++;
    }
    if (slot > 0) {
      this.device.queue.writeBuffer(this.linesUniformBuf, 0, this.linesUniformCpu, 0, slot * stride);
    }
    return slot;
  }

  private encodeLines(pass: GPURenderPassEncoderLike, isHdr: boolean): void {
    let slot = 0;
    for (let i = 0; i < this.flatLines.length && slot < MAX_LINES_SLOTS; i++) {
      const ls = this.flatLines[i];
      if (!isObjVisible(ls)) continue;
      const rec = this.linesCache.get(ls.id);
      if (!rec || rec.count === 0) continue;
      const blendMode = ls.material.transparent ? (ls.material.blending === AdditiveBlending ? 2 : 1) : 0;
      pass.setPipeline(this.getLinesPipeline(blendMode, this.msaaSamples, isHdr));
      pass.setBindGroup(0, this.linesBg, [slot * this.uAlign]);
      pass.setVertexBuffer(0, rec.buf);
      pass.draw(rec.count);
      this.info.render.calls++;
      slot++;
    }
  }

  private preparePoints(camera: Camera, toneMapInline: boolean): number {
    let slot = 0;
    const stride = this.uAlign / 4;
    for (let i = 0; i < this.flatPoints.length && slot < MAX_POINTS_SLOTS; i++) {
      const pts = this.flatPoints[i];
      if (!isObjVisible(pts)) continue;
      const posAttr = pts.geometry.attributes.position;
      if (!posAttr || posAttr.count === 0) continue;
      const colAttr = pts.geometry.attributes.color;
      const sizeAttr = pts.geometry.attributes.size;
      const alphaAttr = pts.geometry.attributes.alpha;

      let rec = this.pointsCache.get(pts.id);
      if (!rec || rec.cap < posAttr.count) {
        if (rec) rec.buf.destroy();
        const cap = Math.max(posAttr.count, 64);
        rec = {
          buf: this.device.createBuffer({ size: cap * 32, usage: GPU_BUFFER_USAGE.VERTEX | GPU_BUFFER_USAGE.COPY_DST }),
          cap,
          posVer: -1, colVer: -1, sizeVer: -1, alphaVer: -1,
          count: 0,
          scratch: new Float32Array(cap * 8),
        };
        this.pointsCache.set(pts.id, rec);
      }
      const pV = posAttr.version, cV = colAttr?.version ?? 0, sV = sizeAttr?.version ?? 0, aV = alphaAttr?.version ?? 0;
      if (rec.posVer !== pV || rec.colVer !== cV || rec.sizeVer !== sV || rec.alphaVer !== aV) {
        const pa = posAttr.array;
        const ca = colAttr?.array;
        const sa = sizeAttr?.array;
        const aa = alphaAttr?.array;
        const sc = rec.scratch;
        for (let k = 0; k < posAttr.count; k++) {
          const o = k * 8, k3 = k * 3;
          sc[o] = pa[k3]; sc[o + 1] = pa[k3 + 1]; sc[o + 2] = pa[k3 + 2];
          sc[o + 3] = sa ? sa[k] : 1;
          sc[o + 4] = ca ? ca[k3] : 1; sc[o + 5] = ca ? ca[k3 + 1] : 1; sc[o + 6] = ca ? ca[k3 + 2] : 1;
          sc[o + 7] = aa ? aa[k] : 1;
        }
        this.device.queue.writeBuffer(rec.buf, 0, sc, 0, posAttr.count * 8);
        rec.posVer = pV; rec.colVer = cV; rec.sizeVer = sV; rec.alphaVer = aV;
        rec.count = posAttr.count;
      }

      const o = slot * stride;
      const f = this.pointsUniformCpu;
      f.set(_vpZo, o);
      f.set(camera.matrixWorldInverse.elements, o + 16);
      f.set(pts.matrixWorld.elements, o + 32);
      f[o + 48] = this.pixelRatio; f[o + 49] = 1 / this.fbW; f[o + 50] = 1 / this.fbH; f[o + 51] = this.toneMappingExposure;
      f[o + 52] = toneMapInline ? 1 : 0; f[o + 53] = 0; f[o + 54] = 0; f[o + 55] = 0;
      slot++;
    }
    if (slot > 0) {
      this.device.queue.writeBuffer(this.pointsUniformBuf, 0, this.pointsUniformCpu, 0, slot * stride);
    }
    return slot;
  }

  private encodePoints(pass: GPURenderPassEncoderLike, isHdr: boolean): void {
    let slot = 0;
    for (let i = 0; i < this.flatPoints.length && slot < MAX_POINTS_SLOTS; i++) {
      const pts = this.flatPoints[i];
      if (!isObjVisible(pts)) continue;
      const rec = this.pointsCache.get(pts.id);
      if (!rec || rec.count === 0) continue;
      const blendMode = pts.material.transparent ? (pts.material.blending === AdditiveBlending ? 2 : 1) : 0;
      pass.setPipeline(this.getPointsPipeline(blendMode, this.msaaSamples, isHdr));
      pass.setBindGroup(0, this.pointsBg, [slot * this.uAlign]);
      pass.setVertexBuffer(0, rec.buf);
      pass.draw(6, rec.count);
      this.info.render.calls++;
      this.info.render.triangles += rec.count * 2;
      slot++;
    }
  }

  private encodePostProcess(enc: GPUCommandEncoderLike, camera: Camera, swapView: GPUTextureViewLike): void {
    const queue = this.device.queue;
    const hasAo = Boolean(this.ssaoBg && this.tgtAoW > 0 && this.tgtAoH > 0);
    const bw = Math.max(1, this.fbW >> 2), bh = Math.max(1, this.fbH >> 2);

    // Upload post uniforms (zero per-frame heap allocations)
    if (hasAo) {
      const pCam = camera as PerspectiveCamera;
      const tanH = Math.tan(((pCam.fov || 62) * Math.PI) / 360);
      const aspect = pCam.aspect || (this.fbW / Math.max(1, this.fbH));
      const af = this.ssaoUniformF32;
      const au = this.ssaoUniformU32;
      af[0] = 1 / this.fbW; af[1] = 1 / this.fbH;
      af[2] = this.tgtAoW;  af[3] = this.tgtAoH;
      af[4] = camera.near;  af[5] = camera.far; af[6] = tanH * aspect; af[7] = tanH;
      au[8] = this.postConfig.aoScale < 0.7 ? 6 : 8;
      au[9] = 0; au[10] = 0; au[11] = 0;
      queue.writeBuffer(this.ssaoUniformBuf, 0, this.ssaoUniformRaw);
    }

    const be = this.bloomExtUniformCpu;
    be[0] = 1 / this.fbW; be[1] = 1 / this.fbH; be[2] = 0; be[3] = 0;
    queue.writeBuffer(this.bloomExtUniformBuf, 0, be);

    const blurBuf = this.bloomBlurUniformCpu;
    const stride = this.uAlign / 4;
    blurBuf[0] = 1.5 / bw; blurBuf[1] = 0;
    blurBuf[stride] = 0;   blurBuf[stride + 1] = 1.5 / bh;
    queue.writeBuffer(this.bloomBlurUniformBuf, 0, blurBuf);

    const cu = this.compUniformCpu;
    cu[0] = hasAo ? 1 / this.tgtAoW : 0;
    cu[1] = hasAo ? 1 / this.tgtAoH : 0;
    cu[2] = this.postConfig.bloomStrength;
    cu[3] = hasAo ? 1 : 0;
    cu[4] = this.toneMappingExposure;
    cu[5] = 0; cu[6] = 0; cu[7] = 0;
    queue.writeBuffer(this.compUniformBuf, 0, cu);

    // 1. Compute Pass: SSAO + Bloom Extract + Horizontal Blur + Vertical Blur (ZERO FBO transitions!)
    const cpass = enc.beginComputePass();
    if (hasAo) {
      cpass.setPipeline(this.msaaSamples > 1 ? this.ssaoMsaaPipe : this.ssaoSinglePipe);
      cpass.setBindGroup(0, this.ssaoBg!);
      cpass.dispatchWorkgroups(Math.ceil(this.tgtAoW / 8), Math.ceil(this.tgtAoH / 8));
      this.info.render.calls++;
    }

    cpass.setPipeline(this.bloomExtPipe);
    cpass.setBindGroup(0, this.bloomExtBg!, [0]);
    cpass.dispatchWorkgroups(Math.ceil(bw / 8), Math.ceil(bh / 8));
    this.info.render.calls++;

    cpass.setPipeline(this.bloomBlurPipe);
    cpass.setBindGroup(0, this.bloomBlurHBg!, [0]);
    cpass.dispatchWorkgroups(Math.ceil(bw / 8), Math.ceil(bh / 8));
    this.info.render.calls++;

    cpass.setBindGroup(0, this.bloomBlurVBg!, [this.uAlign]);
    cpass.dispatchWorkgroups(Math.ceil(bw / 8), Math.ceil(bh / 8));
    this.info.render.calls++;
    cpass.end();

    // 2. Final Composite Render Pass straight into Swapchain!
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: swapView,
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    pass.setPipeline(this.compPipe);
    pass.setBindGroup(0, this.compBg!);
    pass.draw(3);
    pass.end();
    this.info.render.calls++;
  }

  dispose(): void {
    this.freeRenderTargets();
    if (this.shadowArrayTex) { this.shadowArrayTex.destroy(); this.shadowArrayTex = null; }
    for (const g of this.geoCache.values()) this.deleteGeoGpu(g);
    this.geoCache.clear();
    for (const t of this.texCache.values()) t.tex.destroy();
    this.texCache.clear();
    this.matBgCache.clear();
    for (const e of this.instCache.values()) this.destroyInstGpu(e);
    this.instCache.clear();
    for (const p of this.pointsCache.values()) p.buf.destroy();
    this.pointsCache.clear();
    for (const l of this.linesCache.values()) l.buf.destroy();
    this.linesCache.clear();

    this.whiteTex?.destroy();
    this.blackTex?.destroy();
    this.clearTex?.destroy();
    this.dummyShadowTex?.destroy();
    this.sceneUniformBuf?.destroy();
    this.drawStorageBuf?.destroy();
    this.boneStorageBuf?.destroy();
    this.shadowPassUniformBuf?.destroy();
    this.cullParamsBuf?.destroy();
    this.skyUniformBuf?.destroy();
    this.pointsUniformBuf?.destroy();
    this.linesUniformBuf?.destroy();
    this.ssaoUniformBuf?.destroy();
    this.bloomExtUniformBuf?.destroy();
    this.bloomBlurUniformBuf?.destroy();
    this.compUniformBuf?.destroy();
    this.device.destroy();
  }
}
