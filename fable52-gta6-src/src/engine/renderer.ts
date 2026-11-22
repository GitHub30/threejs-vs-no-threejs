import { BufferAttribute, BufferGeometry } from './geometry';
import type { EngineRenderer, PostConfig, RendererOptions, ShadowConfig } from './gpuTypes';
import {
  Material,
  ShaderMaterial,
  Texture,
} from './materials';
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
  BLOOM_BLUR_FRAG,
  BLOOM_EXTRACT_FRAG,
  COMPOSITE_FRAG,
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
  FULLSCREEN_VERT,
  LINES_FRAG,
  LINES_VERT,
  MAX_BONES,
  MAX_CASCADES,
  PBR_FRAG,
  POINTS_FRAG,
  POINTS_VERT,
  SHADOW_FRAG,
  SKY_FRAG,
  SKY_VERT,
  SSAO_FRAG,
  buildPbrVert,
  buildShadowVert,
} from './shaders';

interface ProgramRec {
  program: WebGLProgram;
  uniforms: Map<string, WebGLUniformLocation | null>;
}

interface AttrGpuEntry {
  name: string;
  loc: number;
  buf: WebGLBuffer;
  attr: BufferAttribute;
  version: number;
  byteLength: number;
}

interface GeoGpu {
  vao: WebGLVertexArrayObject;
  attrList: AttrGpuEntry[];
  buffers: Map<string, AttrGpuEntry>;
  indexBuf: WebGLBuffer | null;
  indexAttr: BufferAttribute | null;
  indexVersion: number;
  indexByteLength: number;
  indexType: number;
  indexCount: number;
  vertexCount: number;
  layoutVersion: number;
  disposedVersion: number;
}

interface InstGpu {
  vao: WebGLVertexArrayObject;
  geoId: number;
  geoDisp: number;
  geoLayout: number;
  matBuf: WebGLBuffer;
  colBuf: WebGLBuffer | null;
  spheres: Float32Array;
  spheresVer: number;
  spheresGeoId: number;
  spheresWorldX: number;
  spheresWorldY: number;
  spheresWorldZ: number;
  scratchMat: Float32Array;
  scratchCol: Float32Array | null;
}

interface TexGpu {
  glTex: WebGLTexture;
  version: number;
  disposedVersion: number;
}

interface DrawItem {
  obj: Mesh;
  geo: BufferGeometry;
  mat: Material;
  start: number;
  count: number;
  mode: DrawMode;
  modeId: number; // 0 static, 1 instanced, 2 skinned
  sortKey: number;
  depth: number;
  // Cached world bounding sphere (reused across main pass + all shadow cascades!)
  sx: number;
  sy: number;
  sz: number;
  sr: number;
}

// Module-level zero-allocation scratch buffers & hoisted comparators
const _vp = new Float32Array(16);
const _camPlanes = new Float32Array(24);
const _shPlanes = new Float32Array(24);
const _lightView = new Float32Array(16);
const _lightProj = new Float32Array(16);
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
const _instPrepOut = { vao: null as WebGLVertexArrayObject | null, count: 0 };

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

function glTypeForArray(gl: WebGL2RenderingContext, arr: ArrayBufferView): number {
  if (arr instanceof Float32Array) return gl.FLOAT;
  if (arr instanceof Uint16Array) return gl.UNSIGNED_SHORT;
  if (arr instanceof Int16Array) return gl.SHORT;
  if (arr instanceof Uint8Array) return gl.UNSIGNED_BYTE;
  if (arr instanceof Int8Array) return gl.BYTE;
  if (arr instanceof Uint32Array) return gl.UNSIGNED_INT;
  return gl.FLOAT;
}

export class WebGLRenderer implements EngineRenderer {
  readonly backend = 'webgl2' as const;
  readonly domElement: HTMLCanvasElement;
  readonly gl: WebGL2RenderingContext;
  readonly capabilities: { maxTextureSize: number; maxAnisotropy: number; maxSamples: number };

  shadowMap = { enabled: true, type: 2 };
  toneMapping = ACESFilmicToneMapping;
  toneMappingExposure = 0.95;

  postConfig: PostConfig = { enabled: false, aoScale: 0, bloomStrength: 0.32, reflections: false };
  shadowConfig: ShadowConfig = { enabled: true, mapSize: 1024, cascades: null };

  info = {
    render: { calls: 0, triangles: 0 },
    memory: { geometries: 0, textures: 0 },
    programs: [] as ProgramRec[],
  };

  private pixelRatio = 1;
  private cssW = 1;
  private cssH = 1;
  private fbW = 1;
  private fbH = 1;
  private anisoExt: EXT_texture_filter_anisotropic | null = null;
  private hasHalfFloatFbo = false;
  private defaultMsaa = false;
  private msaaActive = false;

  // Pre-compiled shader programs (13 tiny programs total vs Three.js 30-45!)
  private pbrProgs!: Record<DrawMode, ProgramRec>;
  private pbrProgList: ProgramRec[] = [];
  private shadowProgs!: Record<DrawMode, ProgramRec>;
  private skyProg!: ProgramRec;
  private pointsProg!: ProgramRec;
  private linesProg!: ProgramRec;
  private ssaoProg!: ProgramRec;
  private bloomExtractProg!: ProgramRec;
  private bloomBlurProg!: ProgramRec;
  private compositeProg!: ProgramRec;
  private emptyVao!: WebGLVertexArrayObject;

  // 1x1 fallback textures (eliminates conditional sampler recompilation)
  private whiteTex!: WebGLTexture;
  private blackTex!: WebGLTexture;
  private crmDefaultTex!: WebGLTexture;
  private clearTex!: WebGLTexture;
  private dummyShadowArray!: WebGLTexture;

  // GPU resource pools
  private geoCache = new Map<number, GeoGpu>();
  private texCache = new Map<number, TexGpu>();
  private instBufCache = new Map<number, InstGpu>();

  // Shadow map array FBO
  private shadowFbo: WebGLFramebuffer | null = null;
  private shadowArrayTex: WebGLTexture | null = null;
  private shadowTexSize = 0;
  private shadowLayers = 0;
  private shadowMatsFlat = new Float32Array(MAX_CASCADES * 16);
  private cascadeSplits = new Float32Array(MAX_CASCADES);
  private cascadeBiases = new Float32Array(MAX_CASCADES);
  private cascadeDepthBiases = new Float32Array(MAX_CASCADES);
  private shadowFocusX = 0;
  private shadowFocusZ = 0;
  private shadowLowRange2 = Infinity;

  // Post-processing FBOs (with optional 4x MSAA renderbuffer resolve + checkFramebufferStatus fallback)
  private msaaFbo: WebGLFramebuffer | null = null;
  private msaaColorRb: WebGLRenderbuffer | null = null;
  private msaaDepthRb: WebGLRenderbuffer | null = null;
  private sceneFbo: WebGLFramebuffer | null = null;
  private sceneColorTex: WebGLTexture | null = null;
  private sceneDepthTex: WebGLTexture | null = null;
  private aoFbo: WebGLFramebuffer | null = null;
  private aoTex: WebGLTexture | null = null;
  private bloomFboA: WebGLFramebuffer | null = null;
  private bloomTexA: WebGLTexture | null = null;
  private bloomFboB: WebGLFramebuffer | null = null;
  private bloomTexB: WebGLTexture | null = null;
  private postW = 0;
  private postH = 0;
  private postAoW = 0;
  private postAoH = 0;

  // Persistent flattened scene lists (rebuilt ONLY when scene._structureVersion changes!)
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

  // Reusable per-frame draw buckets (zero per-frame heap allocation)
  private opaqueQueue: DrawItem[] = [];
  private transparentQueue: DrawItem[] = [];
  private shadowQueue: DrawItem[] = [];
  private drawPool: DrawItem[] = [];
  private poolIdx = 0;

  // Cached sky colors for analytical IBL reflections
  private skyTop = [0.18, 0.45, 0.8];
  private skyHor = [0.72, 0.85, 0.94];

  // WebGL state shadowing cache
  private curProg: WebGLProgram | null = null;
  private curVao: WebGLVertexArrayObject | null = null;
  private curSide = -1;
  private curBlend = -1;
  private curDepthWrite: boolean | null = null;
  private curDepthTest: boolean | null = null;
  private curPolyOffset: boolean | null = null;
  private curPolyFactor = NaN;
  private curPolyUnits = NaN;
  private curAlphaToCov: boolean | null = null;
  private curTex = new Array<WebGLTexture | null>(10).fill(null);

  constructor(opts: RendererOptions) {
    this.domElement = opts.canvas;
    const gl = opts.canvas.getContext('webgl2', {
      antialias: opts.antialias ?? true,
      alpha: false,
      depth: true,
      stencil: false,
      powerPreference: opts.powerPreference ?? 'high-performance',
      preserveDrawingBuffer: false,
    });
    if (!gl) throw new Error('WebGL2 is required');
    this.gl = gl;

    this.anisoExt = gl.getExtension('EXT_texture_filter_anisotropic');
    const cbf = gl.getExtension('EXT_color_buffer_float');
    const cbhf = gl.getExtension('EXT_color_buffer_half_float');
    this.hasHalfFloatFbo = Boolean(cbf || cbhf);
    gl.getExtension('OES_texture_float_linear');

    this.capabilities = {
      maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE) as number,
      maxAnisotropy: this.anisoExt ? (gl.getParameter(this.anisoExt.MAX_TEXTURE_MAX_ANISOTROPY_EXT) as number) : 1,
      maxSamples: (gl.getParameter(gl.MAX_SAMPLES) as number) || 0,
    };
    this.defaultMsaa = ((gl.getParameter(gl.SAMPLES) as number) || 0) > 0;

    // Set safe global defaults for disabled vertex attributes ONCE
    gl.vertexAttrib3f(1, 0, 1, 0);
    gl.vertexAttrib4f(4, 1, 1, 1, 1);
    gl.vertexAttrib3f(11, 1, 1, 1);

    this.emptyVao = gl.createVertexArray()!;
    this.createFallbackTextures();
    this.compileCorePrograms();

    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);
    gl.clearColor(0.5, 0.7, 0.9, 1.0);
  }

  getContext(): WebGL2RenderingContext {
    return this.gl;
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
    }
    if (updateStyle) {
      this.domElement.style.width = `${this.cssW}px`;
      this.domElement.style.height = `${this.cssH}px`;
    }
  }

  private createFallbackTextures(): void {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + 9);
    const make1x1 = (rgba: [number, number, number, number]) => {
      const t = gl.createTexture()!;
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(rgba));
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      return t;
    };
    this.whiteTex = make1x1([255, 255, 255, 255]);
    this.blackTex = make1x1([0, 0, 0, 255]);
    this.crmDefaultTex = make1x1([255, 255, 255, 255]);
    this.clearTex = make1x1([0, 0, 0, 0]);

    const ds = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, ds);
    gl.texStorage3D(gl.TEXTURE_2D_ARRAY, 1, gl.DEPTH_COMPONENT24, 1, 1, 1);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_COMPARE_MODE, gl.COMPARE_REF_TO_TEXTURE);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_COMPARE_FUNC, gl.LEQUAL);
    this.dummyShadowArray = ds;
    this.curTex.fill(null);
  }

  private linkProgram(vsSrc: string, fsSrc: string): ProgramRec {
    const gl = this.gl;
    const compile = (type: number, src: string) => {
      const sh = gl.createShader(type)!;
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        const log = gl.getShaderInfoLog(sh);
        gl.deleteShader(sh);
        throw new Error(`Shader compile failed: ${log}\n${src.slice(0, 600)}`);
      }
      return sh;
    };
    const vs = compile(gl.VERTEX_SHADER, vsSrc);
    const fs = compile(gl.FRAGMENT_SHADER, fsSrc);
    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(prog);
      gl.deleteProgram(prog);
      throw new Error(`Program link failed: ${log}`);
    }
    const rec: ProgramRec = { program: prog, uniforms: new Map() };
    this.info.programs.push(rec);
    return rec;
  }

  private uloc(p: ProgramRec, name: string): WebGLUniformLocation | null {
    let loc = p.uniforms.get(name);
    if (loc === undefined) {
      loc = this.gl.getUniformLocation(p.program, name);
      p.uniforms.set(name, loc);
    }
    return loc;
  }

  private compileCorePrograms(): void {
    this.pbrProgs = {
      static: this.linkProgram(buildPbrVert('static'), PBR_FRAG),
      instanced: this.linkProgram(buildPbrVert('instanced'), PBR_FRAG),
      skinned: this.linkProgram(buildPbrVert('skinned'), PBR_FRAG),
    };
    this.pbrProgList = [this.pbrProgs.static, this.pbrProgs.instanced, this.pbrProgs.skinned];
    this.shadowProgs = {
      static: this.linkProgram(buildShadowVert('static'), SHADOW_FRAG),
      instanced: this.linkProgram(buildShadowVert('instanced'), SHADOW_FRAG),
      skinned: this.linkProgram(buildShadowVert('skinned'), SHADOW_FRAG),
    };
    this.skyProg = this.linkProgram(SKY_VERT, SKY_FRAG);
    this.pointsProg = this.linkProgram(POINTS_VERT, POINTS_FRAG);
    this.linesProg = this.linkProgram(LINES_VERT, LINES_FRAG);
    this.ssaoProg = this.linkProgram(FULLSCREEN_VERT, SSAO_FRAG);
    this.bloomExtractProg = this.linkProgram(FULLSCREEN_VERT, BLOOM_EXTRACT_FRAG);
    this.bloomBlurProg = this.linkProgram(FULLSCREEN_VERT, BLOOM_BLUR_FRAG);
    this.compositeProg = this.linkProgram(FULLSCREEN_VERT, COMPOSITE_FRAG);

    const gl = this.gl;
    for (let i = 0; i < this.pbrProgList.length; i++) {
      const p = this.pbrProgList[i];
      gl.useProgram(p.program);
      gl.uniform1i(this.uloc(p, 'uMap'), 0);
      gl.uniform1i(this.uloc(p, 'uEmissiveMap'), 1);
      gl.uniform1i(this.uloc(p, 'uCrmMap'), 2);
      gl.uniform1i(this.uloc(p, 'uBumpAoMap'), 3);
      gl.uniform1i(this.uloc(p, 'uPersonMask'), 4);
      gl.uniform1i(this.uloc(p, 'uPersonOver'), 5);
      gl.uniform1i(this.uloc(p, 'uShadowMap'), 6);
    }
    for (const p of [this.shadowProgs.static, this.shadowProgs.instanced, this.shadowProgs.skinned]) {
      gl.useProgram(p.program);
      gl.uniform1i(this.uloc(p, 'uMap'), 0);
    }
    gl.useProgram(this.ssaoProg.program);
    gl.uniform1i(this.uloc(this.ssaoProg, 'uDepth'), 0);
    gl.useProgram(this.bloomExtractProg.program);
    gl.uniform1i(this.uloc(this.bloomExtractProg, 'uSrc'), 0);
    gl.useProgram(this.bloomBlurProg.program);
    gl.uniform1i(this.uloc(this.bloomBlurProg, 'uSrc'), 0);
    gl.useProgram(this.compositeProg.program);
    gl.uniform1i(this.uloc(this.compositeProg, 'uScene'), 0);
    gl.uniform1i(this.uloc(this.compositeProg, 'uBloom'), 1);
    gl.uniform1i(this.uloc(this.compositeProg, 'uAo'), 2);
    gl.useProgram(null);
  }

  // =========================================================================
  // WebGL State Shadowing Helpers
  // =========================================================================

  private useProg(p: ProgramRec): void {
    if (this.curProg !== p.program) {
      this.gl.useProgram(p.program);
      this.curProg = p.program;
    }
  }

  private bindVao(vao: WebGLVertexArrayObject | null): void {
    if (this.curVao !== vao) {
      this.gl.bindVertexArray(vao);
      this.curVao = vao;
    }
  }

  private bindTex2D(unit: number, tex: WebGLTexture): void {
    if (this.curTex[unit] !== tex) {
      this.gl.activeTexture(this.gl.TEXTURE0 + unit);
      this.gl.bindTexture(this.gl.TEXTURE_2D, tex);
      this.curTex[unit] = tex;
    }
  }

  private setSide(side: number): void {
    if (this.curSide === side) return;
    const gl = this.gl;
    if (side === DoubleSide) {
      gl.disable(gl.CULL_FACE);
    } else {
      gl.enable(gl.CULL_FACE);
      gl.cullFace(side === BackSide ? gl.FRONT : gl.BACK);
    }
    this.curSide = side;
  }

  private setBlend(transparent: boolean, blending: number): void {
    const key = transparent ? (blending === AdditiveBlending ? 2 : 1) : 0;
    if (this.curBlend === key) return;
    const gl = this.gl;
    if (key === 0) {
      gl.disable(gl.BLEND);
    } else {
      gl.enable(gl.BLEND);
      if (key === 2) gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
      else gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    }
    this.curBlend = key;
  }

  private setDepthWrite(write: boolean): void {
    if (this.curDepthWrite !== write) {
      this.gl.depthMask(write);
      this.curDepthWrite = write;
    }
  }

  private setDepthTest(test: boolean): void {
    if (this.curDepthTest !== test) {
      if (test) this.gl.enable(this.gl.DEPTH_TEST);
      else this.gl.disable(this.gl.DEPTH_TEST);
      this.curDepthTest = test;
    }
  }

  private setPolygonOffset(enabled: boolean, factor = 0, units = 0): void {
    const gl = this.gl;
    if (enabled !== this.curPolyOffset) {
      if (enabled) gl.enable(gl.POLYGON_OFFSET_FILL);
      else gl.disable(gl.POLYGON_OFFSET_FILL);
      this.curPolyOffset = enabled;
    }
    if (enabled && (factor !== this.curPolyFactor || units !== this.curPolyUnits)) {
      gl.polygonOffset(factor, units);
      this.curPolyFactor = factor;
      this.curPolyUnits = units;
    }
  }

  private setAlphaToCoverage(enabled: boolean): void {
    const want = enabled && this.msaaActive;
    if (this.curAlphaToCov !== want) {
      if (want) this.gl.enable(this.gl.SAMPLE_ALPHA_TO_COVERAGE);
      else this.gl.disable(this.gl.SAMPLE_ALPHA_TO_COVERAGE);
      this.curAlphaToCov = want;
    }
  }

  private resetStateCache(): void {
    this.curProg = null;
    this.curVao = null;
    this.curSide = -1;
    this.curBlend = -1;
    this.curDepthWrite = null;
    this.curDepthTest = null;
    this.curPolyOffset = null;
    this.curPolyFactor = NaN;
    this.curPolyUnits = NaN;
    this.curAlphaToCov = null;
    this.curTex.fill(null);
  }

  // =========================================================================
  // GPU Resource Sync (Zero-Allocation Fast Paths + Layout Tracking)
  // =========================================================================

  private syncTexture(tex: Texture | null, fallback: WebGLTexture): WebGLTexture {
    if (!tex) return fallback;
    const gl = this.gl;
    let rec = this.texCache.get(tex.id);
    if (rec && rec.disposedVersion !== tex._disposedVersion) {
      gl.deleteTexture(rec.glTex);
      this.texCache.delete(tex.id);
      rec = undefined;
    }
    if (!rec) {
      const glTex = gl.createTexture()!;
      rec = { glTex, version: -1, disposedVersion: tex._disposedVersion };
      this.texCache.set(tex.id, rec);
      tex._onDispose = (t) => {
        const r = this.texCache.get(t.id);
        if (r) {
          gl.deleteTexture(r.glTex);
          this.texCache.delete(t.id);
          this.info.memory.textures = this.texCache.size;
        }
      };
      this.info.memory.textures = this.texCache.size;
    }
    if (rec.version !== tex.version) {
      gl.activeTexture(gl.TEXTURE0 + 9);
      gl.bindTexture(gl.TEXTURE_2D, rec.glTex);
      this.curTex[9] = rec.glTex;

      const srgb = tex.colorSpace === SRGBColorSpace;
      const internalFmt = srgb ? gl.SRGB8_ALPHA8 : gl.RGBA8;

      if (tex.isDataTexture && tex.data) {
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, tex.flipY ? 1 : 0);
        gl.texImage2D(gl.TEXTURE_2D, 0, internalFmt, tex.width, tex.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, tex.data);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        rec.version = tex.version;
      } else if (tex.image) {
        const img = tex.image;
        const w = (img as { width: number }).width;
        const h = (img as { height: number }).height;
        if (w > 0 && h > 0) {
          gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, tex.flipY ? 1 : 0);
          gl.texImage2D(gl.TEXTURE_2D, 0, internalFmt, gl.RGBA, gl.UNSIGNED_BYTE, img as TexImageSource);
          gl.generateMipmap(gl.TEXTURE_2D);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
          if (this.anisoExt && tex.anisotropy > 1) {
            gl.texParameterf(gl.TEXTURE_2D, this.anisoExt.TEXTURE_MAX_ANISOTROPY_EXT, Math.min(tex.anisotropy, this.capabilities.maxAnisotropy));
          }
          rec.version = tex.version;
        } else {
          return fallback;
        }
      } else {
        return fallback;
      }
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, tex.wrapS === RepeatWrapping ? gl.REPEAT : gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, tex.wrapT === RepeatWrapping ? gl.REPEAT : gl.CLAMP_TO_EDGE);
    }
    return rec.glTex;
  }

  private deleteGeoGpu(rec: GeoGpu): void {
    const gl = this.gl;
    gl.deleteVertexArray(rec.vao);
    for (let i = 0; i < rec.attrList.length; i++) gl.deleteBuffer(rec.attrList[i].buf);
    if (rec.indexBuf) gl.deleteBuffer(rec.indexBuf);
  }

  private bindGeoAttributes(gl: WebGL2RenderingContext, geo: BufferGeometry, rec: GeoGpu, dynamicHint: boolean): void {
    const usage = dynamicHint ? gl.DYNAMIC_DRAW : gl.STATIC_DRAW;
    const layoutChanged = rec.layoutVersion !== geo._layoutVersion;

    const mapAttr = (name: string, loc: number) => {
      const attr: BufferAttribute | undefined = geo.attributes[name];
      if (!attr) {
        if (layoutChanged) gl.disableVertexAttribArray(loc);
        return;
      }
      let entry = rec.buffers.get(name);
      let isNew = false;
      if (!entry) {
        const buf = gl.createBuffer()!;
        entry = { name, loc, buf, attr, version: -1, byteLength: 0 };
        rec.buffers.set(name, entry);
        rec.attrList.push(entry);
        isNew = true;
      }
      const attrReplaced = entry.attr !== attr;
      if (isNew || attrReplaced || entry.version !== attr.version) {
        gl.bindBuffer(gl.ARRAY_BUFFER, entry.buf);
        const bl = attr.array.byteLength;
        if (dynamicHint && !attrReplaced && entry.byteLength === bl && bl > 0) {
          gl.bufferSubData(gl.ARRAY_BUFFER, 0, attr.array);
        } else {
          gl.bufferData(gl.ARRAY_BUFFER, attr.array, usage);
          entry.byteLength = bl;
        }
        entry.attr = attr;
        entry.version = attr.version;
        if (isNew || layoutChanged || attrReplaced) {
          gl.enableVertexAttribArray(loc);
          gl.vertexAttribPointer(loc, attr.itemSize, glTypeForArray(gl, attr.array), attr.normalized, 0, 0);
        }
      } else if (layoutChanged) {
        gl.bindBuffer(gl.ARRAY_BUFFER, entry.buf);
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, attr.itemSize, glTypeForArray(gl, attr.array), attr.normalized, 0, 0);
      }
    };

    mapAttr('position', 0);
    mapAttr('normal', 1);
    mapAttr(geo.attributes.size ? 'size' : 'uv', 2);
    mapAttr(geo.attributes.alpha ? 'alpha' : 'uv1', 3);
    mapAttr('color', 4);
    mapAttr('skinIndex', 5);
    mapAttr('skinWeight', 6);

    if (layoutChanged) {
      let write = 0;
      for (let i = 0; i < rec.attrList.length; i++) {
        const e = rec.attrList[i];
        if (geo.attributes[e.name]) {
          rec.attrList[write++] = e;
        } else {
          gl.deleteBuffer(e.buf);
          rec.buffers.delete(e.name);
        }
      }
      rec.attrList.length = write;
    }

    if (geo.index) {
      if (!rec.indexBuf) rec.indexBuf = gl.createBuffer()!;
      const idxReplaced = rec.indexAttr !== geo.index;
      if (idxReplaced || rec.indexVersion !== geo.index.version || layoutChanged) {
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, rec.indexBuf);
        if (idxReplaced || rec.indexVersion !== geo.index.version) {
          const bl = geo.index.array.byteLength;
          if (dynamicHint && !idxReplaced && rec.indexByteLength === bl && bl > 0) {
            gl.bufferSubData(gl.ELEMENT_ARRAY_BUFFER, 0, geo.index.array);
          } else {
            gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, geo.index.array, usage);
            rec.indexByteLength = bl;
          }
          rec.indexAttr = geo.index;
          rec.indexVersion = geo.index.version;
        }
      }
      rec.indexType = geo.index.array instanceof Uint32Array ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT;
      rec.indexCount = geo.index.count;
    } else {
      if (rec.indexBuf && layoutChanged) {
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
        gl.deleteBuffer(rec.indexBuf);
        rec.indexBuf = null;
        rec.indexVersion = -1;
        rec.indexByteLength = 0;
      }
      rec.indexAttr = null;
      rec.indexCount = 0;
    }
    rec.vertexCount = geo.attributes.position ? geo.attributes.position.count : 0;
    rec.layoutVersion = geo._layoutVersion;
  }

  private syncGeometry(geo: BufferGeometry, dynamicHint = false): GeoGpu {
    const gl = this.gl;
    let rec = this.geoCache.get(geo.id);
    if (rec && rec.disposedVersion !== geo._disposedVersion) {
      this.deleteGeoGpu(rec);
      this.geoCache.delete(geo.id);
      rec = undefined;
    }
    if (!rec) {
      const vao = gl.createVertexArray()!;
      rec = {
        vao,
        attrList: [],
        buffers: new Map(),
        indexBuf: null,
        indexAttr: null,
        indexVersion: -1,
        indexByteLength: 0,
        indexType: gl.UNSIGNED_SHORT,
        indexCount: 0,
        vertexCount: 0,
        layoutVersion: -1,
        disposedVersion: geo._disposedVersion,
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
      this.bindVao(vao);
      this.bindGeoAttributes(gl, geo, rec, dynamicHint);
      return rec;
    }
    let dirty = rec.layoutVersion !== geo._layoutVersion || (geo.index !== null && rec.indexVersion !== geo.index.version);
    if (!dirty) {
      const list = rec.attrList;
      for (let i = 0; i < list.length; i++) {
        const e = list[i];
        const a = geo.attributes[e.name];
        if (!a || e.attr !== a || e.version !== a.version) {
          dirty = true;
          break;
        }
      }
    }
    if (dirty) {
      this.bindVao(rec.vao);
      this.bindGeoAttributes(gl, geo, rec, dynamicHint);
    }
    return rec;
  }

  /**
   * Frustum-culled InstancedMesh preparation with STREAM_DRAW buffer orphaning:
   * - Zero .subarray() allocations (uses WebGL2 bufferData srcOffset + length)
   * - Zero cross-cascade GPU pipeline stall hazard
   * - Includes mesh.matrixWorld in instance spheres and handles late setColorAt()
   */
  private prepareInstancedDraw(
    mesh: InstancedMesh,
    geoRec: GeoGpu,
    planes: Float32Array,
  ): { vao: WebGLVertexArrayObject | null; count: number } {
    const gl = this.gl;
    const total = mesh.count;
    let entry = this.instBufCache.get(mesh.id);
    if (entry && (entry.geoId !== mesh.geometry.id || entry.geoDisp !== mesh.geometry._disposedVersion || entry.geoLayout !== mesh.geometry._layoutVersion)) {
      gl.deleteVertexArray(entry.vao);
      gl.deleteBuffer(entry.matBuf);
      if (entry.colBuf) gl.deleteBuffer(entry.colBuf);
      this.instBufCache.delete(mesh.id);
      entry = undefined;
    }
    if (!entry) {
      const vao = gl.createVertexArray()!;
      const matBuf = gl.createBuffer()!;
      entry = {
        vao,
        geoId: mesh.geometry.id,
        geoDisp: mesh.geometry._disposedVersion,
        geoLayout: mesh.geometry._layoutVersion,
        matBuf,
        colBuf: null,
        spheres: new Float32Array(total * 4),
        spheresVer: -1,
        spheresGeoId: -1,
        spheresWorldX: NaN,
        spheresWorldY: NaN,
        spheresWorldZ: NaN,
        scratchMat: new Float32Array(total * 16),
        scratchCol: null,
      };
      this.instBufCache.set(mesh.id, entry);
      mesh._onDispose = (im) => {
        const e = this.instBufCache.get(im.id);
        if (e) {
          gl.deleteVertexArray(e.vao);
          gl.deleteBuffer(e.matBuf);
          if (e.colBuf) gl.deleteBuffer(e.colBuf);
          this.instBufCache.delete(im.id);
        }
      };

      this.bindVao(vao);
      const geo = mesh.geometry;
      const attach = (name: string, loc: number) => {
        const attr = geo.attributes[name];
        const b = geoRec.buffers.get(name);
        if (!attr || !b) { gl.disableVertexAttribArray(loc); return; }
        gl.bindBuffer(gl.ARRAY_BUFFER, b.buf);
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, attr.itemSize, glTypeForArray(gl, attr.array), attr.normalized, 0, 0);
      };
      attach('position', 0);
      attach('normal', 1);
      attach('uv', 2);
      attach('uv1', 3);
      attach('color', 4);
      if (geoRec.indexBuf) gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, geoRec.indexBuf);

      gl.bindBuffer(gl.ARRAY_BUFFER, matBuf);
      gl.bufferData(gl.ARRAY_BUFFER, total * 64, gl.STREAM_DRAW);
      for (let col = 0; col < 4; col++) {
        const loc = 7 + col;
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, 4, gl.FLOAT, false, 64, col * 16);
        gl.vertexAttribDivisor(loc, 1);
      }
      gl.disableVertexAttribArray(11);
    }

    // Lazily attach color buffer even if setColorAt() was called after first draw
    if (mesh.instanceColor && !entry.colBuf) {
      entry.colBuf = gl.createBuffer()!;
      entry.scratchCol = new Float32Array(total * 3);
      this.bindVao(entry.vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, entry.colBuf);
      gl.bufferData(gl.ARRAY_BUFFER, total * 12, gl.STREAM_DRAW);
      gl.enableVertexAttribArray(11);
      gl.vertexAttribPointer(11, 3, gl.FLOAT, false, 12, 0);
      gl.vertexAttribDivisor(11, 1);
    }

    const mw = mesh.matrixWorld.elements;
    if (
      entry.spheresVer !== mesh.instanceMatrix.version ||
      entry.spheresGeoId !== mesh.geometry.id ||
      entry.spheresWorldX !== mw[12] ||
      entry.spheresWorldY !== mw[13] ||
      entry.spheresWorldZ !== mw[14]
    ) {
      const bs = mesh.geometry.boundingSphere ?? mesh.geometry.computeBoundingSphere();
      const bx = bs.center.x, by = bs.center.y, bz = bs.center.z, br = bs.radius;
      const src = mesh.instanceMatrix.array;
      const sph = entry.spheres;
      const wSc = Math.sqrt(Math.max(
        mw[0] * mw[0] + mw[1] * mw[1] + mw[2] * mw[2],
        mw[4] * mw[4] + mw[5] * mw[5] + mw[6] * mw[6],
        mw[8] * mw[8] + mw[9] * mw[9] + mw[10] * mw[10],
      ));
      for (let i = 0; i < total; i++) {
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
      entry.spheresVer = mesh.instanceMatrix.version;
      entry.spheresGeoId = mesh.geometry.id;
      entry.spheresWorldX = mw[12];
      entry.spheresWorldY = mw[13];
      entry.spheresWorldZ = mw[14];
    }

    const srcM = mesh.instanceMatrix.array;
    const dstM = entry.scratchMat;
    const srcC = mesh.instanceColor ? mesh.instanceColor.array : null;
    const dstC = entry.scratchCol;
    const sph = entry.spheres;
    let vis = 0;
    for (let i = 0; i < total; i++) {
      const s = i * 4;
      if (!sphereIntersectsFrustum(planes, sph[s], sph[s + 1], sph[s + 2], sph[s + 3])) continue;
      const so = i * 16, doff = vis * 16;
      for (let k = 0; k < 16; k++) dstM[doff + k] = srcM[so + k];
      if (srcC && dstC) {
        const sc = i * 3, dc = vis * 3;
        dstC[dc] = srcC[sc]; dstC[dc + 1] = srcC[sc + 1]; dstC[dc + 2] = srcC[sc + 2];
      }
      vis++;
    }

    if (vis > 0) {
      // Orphan GPU storage via STREAM_DRAW bufferData(srcOffset, length): zero subarray GC + zero cascade pipeline stall!
      gl.bindBuffer(gl.ARRAY_BUFFER, entry.matBuf);
      gl.bufferData(gl.ARRAY_BUFFER, dstM, gl.STREAM_DRAW, 0, vis * 16);
      if (entry.colBuf && dstC) {
        gl.bindBuffer(gl.ARRAY_BUFFER, entry.colBuf);
        gl.bufferData(gl.ARRAY_BUFFER, dstC, gl.STREAM_DRAW, 0, vis * 3);
      }
    }
    _instPrepOut.vao = entry.vao;
    _instPrepOut.count = vis;
    return _instPrepOut;
  }

  // =========================================================================
  // Shadow Map Array & Post-Processing FBO Management (Zero State Clobber / Zero Leak)
  // =========================================================================

  private ensureShadowTargets(size: number, layers: number): void {
    if (this.shadowFbo && this.shadowTexSize === size && this.shadowLayers === layers) return;
    const gl = this.gl;
    if (this.shadowFbo) gl.deleteFramebuffer(this.shadowFbo);
    if (this.shadowArrayTex) gl.deleteTexture(this.shadowArrayTex);

    gl.activeTexture(gl.TEXTURE0 + 9);
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, tex);
    gl.texStorage3D(gl.TEXTURE_2D_ARRAY, 1, gl.DEPTH_COMPONENT24, size, size, layers);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_COMPARE_MODE, gl.COMPARE_REF_TO_TEXTURE);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_COMPARE_FUNC, gl.LEQUAL);
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, null);

    this.shadowFbo = gl.createFramebuffer()!;
    this.shadowArrayTex = tex;
    this.shadowTexSize = size;
    this.shadowLayers = layers;
    this.curTex[9] = null;
  }

  private freePostTargets(): void {
    const gl = this.gl;
    if (this.msaaFbo) { gl.deleteFramebuffer(this.msaaFbo); this.msaaFbo = null; }
    if (this.msaaColorRb) { gl.deleteRenderbuffer(this.msaaColorRb); this.msaaColorRb = null; }
    if (this.msaaDepthRb) { gl.deleteRenderbuffer(this.msaaDepthRb); this.msaaDepthRb = null; }
    if (this.sceneFbo) { gl.deleteFramebuffer(this.sceneFbo); this.sceneFbo = null; }
    if (this.sceneColorTex) { gl.deleteTexture(this.sceneColorTex); this.sceneColorTex = null; }
    if (this.sceneDepthTex) { gl.deleteTexture(this.sceneDepthTex); this.sceneDepthTex = null; }
    if (this.aoFbo) { gl.deleteFramebuffer(this.aoFbo); this.aoFbo = null; }
    if (this.aoTex) { gl.deleteTexture(this.aoTex); this.aoTex = null; }
    if (this.bloomFboA) { gl.deleteFramebuffer(this.bloomFboA); this.bloomFboA = null; }
    if (this.bloomTexA) { gl.deleteTexture(this.bloomTexA); this.bloomTexA = null; }
    if (this.bloomFboB) { gl.deleteFramebuffer(this.bloomFboB); this.bloomFboB = null; }
    if (this.bloomTexB) { gl.deleteTexture(this.bloomTexB); this.bloomTexB = null; }
    this.postW = this.postH = this.postAoW = this.postAoH = 0;
    this.msaaActive = false;
  }

  private ensurePostTargets(w: number, h: number, aoScale: number): boolean {
    const aoW = aoScale > 0 ? Math.max(1, Math.round(w * aoScale)) : 0;
    const aoH = aoScale > 0 ? Math.max(1, Math.round(h * aoScale)) : 0;
    if (this.sceneFbo && this.postW === w && this.postH === h && this.postAoW === aoW && this.postAoH === aoH) return true;

    const gl = this.gl;
    this.freePostTargets();
    // Always use scratch unit 9 so we NEVER clobber active texture units 0..6!
    gl.activeTexture(gl.TEXTURE0 + 9);

    const makeColorTex = (tw: number, th: number, internalFmt: number, fmt: number, type: number) => {
      const t = gl.createTexture()!;
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, internalFmt, tw, th, 0, fmt, type, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      return t;
    };

    // Try RGBA16F first, validate checkFramebufferStatus, and fall back to RGBA8 if unsupported
    let hdrInternal = this.hasHalfFloatFbo ? gl.RGBA16F : gl.RGBA8;
    let hdrType = this.hasHalfFloatFbo ? gl.HALF_FLOAT : gl.UNSIGNED_BYTE;

    const buildSceneFbo = (intFmt: number, pxType: number) => {
      this.sceneColorTex = makeColorTex(w, h, intFmt, gl.RGBA, pxType);
      const dTex = gl.createTexture()!;
      gl.bindTexture(gl.TEXTURE_2D, dTex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.DEPTH_COMPONENT24, w, h, 0, gl.DEPTH_COMPONENT, gl.UNSIGNED_INT, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      this.sceneDepthTex = dTex;

      this.sceneFbo = gl.createFramebuffer()!;
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.sceneFbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.sceneColorTex, 0);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, this.sceneDepthTex, 0);
      return gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
    };

    if (!buildSceneFbo(hdrInternal, hdrType)) {
      this.freePostTargets();
      hdrInternal = gl.RGBA8;
      hdrType = gl.UNSIGNED_BYTE;
      if (!buildSceneFbo(hdrInternal, hdrType)) {
        this.freePostTargets();
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        this.curTex.fill(null);
        return false;
      }
    }

    // Spec-compliant MSAA renderbuffer FBO: query supported sample counts for BOTH color and depth formats first!
    let samples = 0;
    if (this.capabilities.maxSamples >= 2) {
      const cList = gl.getInternalformatParameter(gl.RENDERBUFFER, hdrInternal, gl.SAMPLES) as Int32Array | null;
      const dList = gl.getInternalformatParameter(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, gl.SAMPLES) as Int32Array | null;
      if (cList && dList) {
        for (let i = 0; i < cList.length; i++) {
          const s = cList[i];
          if (s >= 2 && s <= 4 && s > samples) {
            for (let j = 0; j < dList.length; j++) {
              if (dList[j] === s) { samples = s; break; }
            }
          }
        }
      }
    }
    if (samples >= 2) {
      const mFbo = gl.createFramebuffer()!;
      const cRb = gl.createRenderbuffer()!;
      const dRb = gl.createRenderbuffer()!;
      gl.bindRenderbuffer(gl.RENDERBUFFER, cRb);
      gl.renderbufferStorageMultisample(gl.RENDERBUFFER, samples, hdrInternal, w, h);
      gl.bindRenderbuffer(gl.RENDERBUFFER, dRb);
      gl.renderbufferStorageMultisample(gl.RENDERBUFFER, samples, gl.DEPTH_COMPONENT24, w, h);
      gl.bindFramebuffer(gl.FRAMEBUFFER, mFbo);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, cRb);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, dRb);
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE) {
        this.msaaFbo = mFbo;
        this.msaaColorRb = cRb;
        this.msaaDepthRb = dRb;
        this.msaaActive = true;
      } else {
        gl.deleteFramebuffer(mFbo);
        gl.deleteRenderbuffer(cRb);
        gl.deleteRenderbuffer(dRb);
        this.msaaActive = false;
      }
      gl.bindRenderbuffer(gl.RENDERBUFFER, null);
    }

    if (aoW > 0 && aoH > 0) {
      this.aoTex = makeColorTex(aoW, aoH, gl.R8, gl.RED, gl.UNSIGNED_BYTE);
      this.aoFbo = gl.createFramebuffer()!;
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.aoFbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.aoTex, 0);
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
        gl.deleteFramebuffer(this.aoFbo);
        gl.deleteTexture(this.aoTex);
        this.aoFbo = null;
        this.aoTex = null;
      }
    }

    const bw = Math.max(1, w >> 2), bh = Math.max(1, h >> 2);
    this.bloomTexA = makeColorTex(bw, bh, hdrInternal, gl.RGBA, hdrType);
    this.bloomFboA = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.bloomFboA);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.bloomTexA, 0);

    this.bloomTexB = makeColorTex(bw, bh, hdrInternal, gl.RGBA, hdrType);
    this.bloomFboB = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.bloomFboB);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.bloomTexB, 0);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
    this.curTex.fill(null);
    this.postW = w;
    this.postH = h;
    this.postAoW = aoW;
    this.postAoH = aoH;
    return true;
  }

  // =========================================================================
  // Persistent Scene Flatten (Eliminates Per-Frame Scene Traversal!)
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
    start: number, count: number, mode: DrawMode, modeId: number, depth: number,
    sx: number, sy: number, sz: number, sr: number,
  ): DrawItem {
    let it = this.drawPool[this.poolIdx];
    const sortKey = (obj.renderOrder + 100) * 1e9 + modeId * 1e7 + mat.id * 1e3 + (geo.id & 1023);
    if (!it) {
      it = { obj, geo, mat, start, count, mode, modeId, sortKey, depth, sx, sy, sz, sr };
      this.drawPool.push(it);
    } else {
      it.obj = obj; it.geo = geo; it.mat = mat; it.start = start; it.count = count;
      it.mode = mode; it.modeId = modeId; it.sortKey = sortKey; it.depth = depth;
      it.sx = sx; it.sy = sy; it.sz = sz; it.sr = sr;
    }
    this.poolIdx++;
    return it;
  }

  // =========================================================================
  // Cascade Matrix Computation (True World-Origin Light-Space Texel Snapping!)
  // =========================================================================

  private computeShadowCascades(camera: Camera): number {
    const sun = this.sunLight;
    if (!this.shadowMap.enabled || !this.shadowConfig.enabled || !sun || sun.intensity < 0.01) return 0;

    // Read world positions from matrixWorld (supports parented lights cleanly)
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

      // True world-origin light-space texel snapping (eliminates sub-texel shadow swimming!)
      const worldLx = _vRight.x * tx + _vRight.y * ty + _vRight.z * tz;
      const worldLy = _vLightUp.x * tx + _vLightUp.y * ty + _vLightUp.z * tz;
      const dx = Math.round(worldLx / texel) * texel - worldLx;
      const dy = Math.round(worldLy / texel) * texel - worldLy;

      const ex = tx + _vSun.x * 240, ey = ty + _vSun.y * 240, ez = tz + _vSun.z * 240;
      const lightNear = 1, lightFar = 560;
      this.buildLightViewMatrix(_lightView, ex, ey, ez);
      _tmpMat4.makeOrthographic(dx - half, dx + half, dy + half, dy - half, lightNear, lightFar);
      _lightProj.set(_tmpMat4.elements);
      multiplyMat4Arrays(this.shadowMatsFlat.subarray(0, 16), _lightProj, _lightView);
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

      // True world-origin light-space texel snapping (fixes no-op snapping bug!)
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
      multiplyMat4Arrays(this.shadowMatsFlat.subarray(c * 16, c * 16 + 16), _lightProj, _lightView);
      this.cascadeBiases[c] = Math.max(0.02, texel * 1.2);
      this.cascadeDepthBiases[c] = (0.03 + texel * 1.5) / (lightFar - lightNear);
    }
    return nCascades;
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

  // =========================================================================
  // Main Frame Render
  // =========================================================================

  render(scene: Scene, camera: Camera): void {
    const gl = this.gl;
    this.info.render.calls = 0;
    this.info.render.triangles = 0;

    scene.updateMatrixWorld(false);
    camera.updateMatrixWorld(false);

    if (this.cachedSceneId !== scene.id || this.cachedStructVer !== scene._structureVersion) {
      this.rebuildSceneLists(scene);
    }

    multiplyMat4Arrays(_vp, camera.projectionMatrix.elements, camera.matrixWorldInverse.elements);
    extractFrustumPlanes(_vp, _camPlanes);
    const camX = camera.matrixWorld.elements[12];
    const camY = camera.matrixWorld.elements[13];
    const camZ = camera.matrixWorld.elements[14];

    if (this.flatSky && this.flatSky.material instanceof ShaderMaterial) {
      const u = this.flatSky.material.uniforms;
      const top = u.top?.value as { r: number; g: number; b: number } | undefined;
      const hor = u.horizon?.value as { r: number; g: number; b: number } | undefined;
      if (top) { this.skyTop[0] = top.r; this.skyTop[1] = top.g; this.skyTop[2] = top.b; }
      if (hor) { this.skyHor[0] = hor.r; this.skyHor[1] = hor.g; this.skyHor[2] = hor.b; }
    }

    this.opaqueQueue.length = 0;
    this.transparentQueue.length = 0;
    this.shadowQueue.length = 0;
    this.poolIdx = 0;

    const meshes = this.flatMeshes;
    const cCount = this.computeShadowCascades(camera);
    const moverFar2 = cCount > 1 ? (this.cascadeSplits[cCount - 2] * 1.15) ** 2 : this.shadowLowRange2;

    for (let i = 0; i < meshes.length; i++) {
      const mesh = meshes[i];
      if (!isObjVisible(mesh)) continue;
      const geo = mesh.geometry;
      if (!geo.attributes.position || geo.attributes.position.count === 0) continue;

      const isInst = (mesh as InstancedMesh).isInstancedMesh === true;
      const isSkin = (mesh as SkinnedMesh).isSkinnedMesh === true;
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
      const mats = Array.isArray(mesh.material) ? mesh.material : null;
      const groups = geo.groups;

      // On single-map low preset, only cast shadows for skinned movers near sun.target (saves skinning + shadow draws!)
      const inShadowRange = !isSkin || (cCount === 1
        ? (sx - this.shadowFocusX) ** 2 + (sz - this.shadowFocusZ) ** 2 < this.shadowLowRange2
        : dist2 < moverFar2 * 1.4);
      const castSh = cCount > 0 && mesh.castShadow && inShadowRange;

      if (isSkin && (inMainFrustum || castSh)) {
        const sm = mesh as SkinnedMesh;
        if (sm.skeleton) {
          invertMat4Array(_meshWorldInv, mw);
          if (sm._hasNonIdentityBind) {
            sm.skeleton.update(_meshWorldInv, sm.bindMatrix.elements, sm.bindMatrixInverse.elements);
          } else {
            sm.skeleton.update(_meshWorldInv);
          }
        }
      }

      const emitGroup = (mat: Material, start: number, count: number) => {
        if (!mat || !mat.visible) return;
        if (inMainFrustum) {
          const item = this.allocDrawItem(mesh, geo, mat, start, count, mode, modeId, dist2, sx, sy, sz, sr);
          if (mat.transparent) this.transparentQueue.push(item);
          else this.opaqueQueue.push(item);
        }
        if (castSh && !mat.transparent) {
          this.shadowQueue.push(this.allocDrawItem(mesh, geo, mat, start, count, mode, modeId, dist2, sx, sy, sz, sr));
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

    this.resetStateCache();

    // 4. Shadow Pass (Hardware TEXTURE_2D_ARRAY CSM + Per-Instance Frustum Culling!)
    if (cCount > 0) {
      this.renderShadowPass(cCount, moverFar2);
    }

    // 5. Main Scene Pass (release post FBOs immediately when disabled to prevent VRAM leak!)
    let usePost = this.postConfig.enabled;
    if (usePost) {
      usePost = this.ensurePostTargets(this.fbW, this.fbH, this.postConfig.aoScale);
    } else if (this.sceneFbo) {
      this.freePostTargets();
    }

    if (usePost) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.msaaActive ? this.msaaFbo : this.sceneFbo);
    } else {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      this.msaaActive = this.defaultMsaa;
    }
    gl.viewport(0, 0, this.fbW, this.fbH);
    this.setDepthTest(true);
    this.setDepthWrite(true);
    this.setBlend(false, 0);
    this.setAlphaToCoverage(false);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    if (this.flatSky && isObjVisible(this.flatSky)) {
      this.renderSky(this.flatSky, !usePost);
    }

    gl.activeTexture(gl.TEXTURE0 + 6);
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, cCount > 0 && this.shadowArrayTex ? this.shadowArrayTex : this.dummyShadowArray);

    this.uploadScenePbrUniforms(scene, camera, cCount);

    this.renderMeshQueue(this.opaqueQueue, scene.environmentIntensity, !usePost);
    this.renderMeshQueue(this.transparentQueue, scene.environmentIntensity, !usePost);
    this.setAlphaToCoverage(false);

    this.renderLines(!usePost);
    this.renderPoints(camera, !usePost);

    if (usePost) {
      // Resolve MSAA renderbuffers into readable HDR color & depth textures cleanly
      if (this.msaaActive && this.msaaFbo && this.sceneFbo) {
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER, this.msaaFbo);
        gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, this.sceneFbo);
        gl.blitFramebuffer(0, 0, this.fbW, this.fbH, 0, 0, this.fbW, this.fbH, gl.COLOR_BUFFER_BIT, gl.NEAREST);
        if (this.aoFbo && this.postAoW > 0) {
          gl.blitFramebuffer(0, 0, this.fbW, this.fbH, 0, 0, this.fbW, this.fbH, gl.DEPTH_BUFFER_BIT, gl.NEAREST);
        }
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
        gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
      }
      this.renderPostProcess(camera);
    }
  }

  private renderShadowPass(cCount: number, moverFar2: number): void {
    const gl = this.gl;
    const size = this.shadowConfig.mapSize;
    this.ensureShadowTargets(size, cCount);

    // Unbind shadow array from sampling unit 6 before attaching to shadow FBO (zero feedback-loop warnings!)
    gl.activeTexture(gl.TEXTURE0 + 6);
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, null);

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.shadowFbo);
    gl.viewport(0, 0, size, size);
    gl.colorMask(false, false, false, false);
    this.setDepthTest(true);
    this.setDepthWrite(true);
    this.setBlend(false, 0);
    this.setAlphaToCoverage(false);
    this.setPolygonOffset(true, 2.2, 3.0);

    const queue = this.shadowQueue;
    queue.sort(sortOpaque);

    for (let c = 0; c < cCount; c++) {
      gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, this.shadowArrayTex!, 0, c);
      gl.clear(gl.DEPTH_BUFFER_BIT);
      const lvp = this.shadowMatsFlat.subarray(c * 16, c * 16 + 16);
      extractFrustumPlanes(lvp, _shPlanes);
      const isFarthest = cCount > 1 && c === cCount - 1;

      let lastMode: DrawMode | null = null;
      let lastAlphaTest = -1;
      for (let i = 0; i < queue.length; i++) {
        const it = queue[i];
        if (isFarthest && it.mode === 'skinned' && it.depth >= moverFar2) continue;

        const mesh = it.obj;
        // Reuse cached world bounding sphere (zero recomputation across 4 cascades!)
        if (mesh.frustumCulled && !sphereIntersectsFrustum(_shPlanes, it.sx, it.sy, it.sz, it.sr)) continue;

        const geoRec = this.syncGeometry(it.geo);
        let instVao: WebGLVertexArrayObject | null = null;
        let instCount = 0;
        if (it.mode === 'instanced') {
          const prep = this.prepareInstancedDraw(mesh as InstancedMesh, geoRec, _shPlanes);
          if (prep.count === 0) continue;
          instVao = prep.vao;
          instCount = prep.count;
        }

        const prog = this.shadowProgs[it.mode];
        if (lastMode !== it.mode) {
          this.useProg(prog);
          gl.uniformMatrix4fv(this.uloc(prog, 'uLightVP'), false, lvp);
          lastMode = it.mode;
          lastAlphaTest = -1;
        }
        this.setSide(it.mat.side === DoubleSide ? DoubleSide : FrontSide);
        gl.uniformMatrix4fv(this.uloc(prog, 'uModel'), false, mesh.matrixWorld.elements);

        const wantAlpha = it.mat.alphaTest > 0 && it.mat.map ? it.mat.alphaTest : 0;
        if (wantAlpha > 0) {
          if (lastAlphaTest !== wantAlpha) {
            gl.uniform1f(this.uloc(prog, 'uAlphaTest'), wantAlpha);
            lastAlphaTest = wantAlpha;
          }
          const m = it.mat.map!;
          gl.uniform4f(this.uloc(prog, 'uUvTransform'), m.repeat.x, m.repeat.y, m.offset.x, m.offset.y);
          this.bindTex2D(0, this.syncTexture(m, this.whiteTex));
        } else if (lastAlphaTest !== 0) {
          gl.uniform1f(this.uloc(prog, 'uAlphaTest'), 0);
          lastAlphaTest = 0;
        }

        if (it.mode === 'skinned') {
          const sk = (mesh as SkinnedMesh).skeleton;
          if (sk) {
            const count = Math.min(sk.bones.length, MAX_BONES) * 16;
            gl.uniformMatrix4fv(this.uloc(prog, 'uBones[0]'), false, sk.boneMatrices, 0, count);
          }
          this.bindVao(geoRec.vao);
          this.drawRange(geoRec, it.start, it.count, 0);
        } else if (it.mode === 'instanced') {
          this.bindVao(instVao);
          this.drawRange(geoRec, it.start, it.count, instCount);
        } else {
          this.bindVao(geoRec.vao);
          this.drawRange(geoRec, it.start, it.count, 0);
        }
      }
    }

    gl.colorMask(true, true, true, true);
    this.setPolygonOffset(false);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  private renderSky(sky: Mesh, toneMapInline: boolean): void {
    const gl = this.gl;
    const mat = sky.material as ShaderMaterial;
    const p = this.skyProg;
    this.useProg(p);
    this.setSide(BackSide);
    this.setDepthWrite(false);
    this.setDepthTest(true);
    this.setBlend(false, 0);
    this.setPolygonOffset(false);

    gl.uniformMatrix4fv(this.uloc(p, 'uViewProj'), false, _vp);
    gl.uniformMatrix4fv(this.uloc(p, 'uModel'), false, sky.matrixWorld.elements);
    const u = mat.uniforms;
    const setC = (name: string) => {
      const c = u[name]?.value as { r: number; g: number; b: number } | undefined;
      if (c) gl.uniform3f(this.uloc(p, name), c.r, c.g, c.b);
    };
    const setV = (name: string) => {
      const v = u[name]?.value as { x: number; y: number; z: number } | undefined;
      if (v) gl.uniform3f(this.uloc(p, name), v.x, v.y, v.z);
    };
    const setF = (name: string) => {
      const v = u[name]?.value as number | undefined;
      if (v !== undefined) gl.uniform1f(this.uloc(p, name), v);
    };
    setC('top'); setC('horizon'); setC('sunColor'); setC('cloudColor');
    setV('sunDir');
    setF('stars'); setF('time'); setF('cover'); setF('moon');
    gl.uniform1i(this.uloc(p, 'uToneMapInline'), toneMapInline ? 1 : 0);
    gl.uniform1f(this.uloc(p, 'uExposure'), this.toneMappingExposure);

    const geoRec = this.syncGeometry(sky.geometry);
    this.bindVao(geoRec.vao);
    const total = sky.geometry.index ? sky.geometry.index.count : sky.geometry.attributes.position.count;
    this.drawRange(geoRec, 0, total, 0);
    this.setDepthWrite(true);
  }

  private uploadScenePbrUniforms(scene: Scene, camera: Camera, cCount: number): void {
    const gl = this.gl;
    const camM = camera.matrixWorld.elements;
    const sun = this.sunLight;
    const hemi = this.hemiLight;
    const spot = this.spotLight;
    const point = this.pointLight;
    const fog = scene.fog;

    if (sun) {
      const sm = sun.matrixWorld.elements, tm = sun.target.matrixWorld.elements;
      _vSun.set(sm[12] - tm[12], sm[13] - tm[13], sm[14] - tm[14]).normalize();
    } else {
      _vSun.set(0.3, 0.9, 0.2).normalize();
    }

    let spX = 0, spY = 0, spZ = 0, spRange = 0, spDx = 0, spDy = 0, spDz = -1, spCos = 1, spPenCos = 1, spDecay = 2;
    let spR = 0, spG = 0, spB = 0;
    if (spot && spot.visible && spot.intensity > 0.01) {
      const sm = spot.matrixWorld.elements, tm = spot.target.matrixWorld.elements;
      spX = sm[12]; spY = sm[13]; spZ = sm[14];
      _vSpotDir.set(tm[12] - spX, tm[13] - spY, tm[14] - spZ).normalize();
      spDx = _vSpotDir.x; spDy = _vSpotDir.y; spDz = _vSpotDir.z;
      spRange = spot.distance > 0 ? spot.distance : 70;
      spCos = Math.cos(spot.angle);
      spPenCos = Math.cos(spot.angle * (1 - spot.penumbra));
      spDecay = spot.decay ?? 2;
      spR = spot.color.r * spot.intensity;
      spG = spot.color.g * spot.intensity;
      spB = spot.color.b * spot.intensity;
    }

    let ptX = 0, ptY = 0, ptZ = 0, ptRange = 0, ptR = 0, ptG = 0, ptB = 0, ptDecay = 1.6;
    if (point && point.visible && point.intensity > 0.01) {
      const pm = point.matrixWorld.elements;
      ptX = pm[12]; ptY = pm[13]; ptZ = pm[14];
      ptRange = point.distance > 0 ? point.distance : 60;
      ptR = point.color.r * point.intensity;
      ptG = point.color.g * point.intensity;
      ptB = point.color.b * point.intensity;
      ptDecay = point.decay ?? 1.6;
    }

    for (let i = 0; i < this.pbrProgList.length; i++) {
      const p = this.pbrProgList[i];
      this.useProg(p);
      gl.uniformMatrix4fv(this.uloc(p, 'uViewProj'), false, _vp);
      gl.uniformMatrix4fv(this.uloc(p, 'uView'), false, camera.matrixWorldInverse.elements);
      gl.uniform3f(this.uloc(p, 'uCamPos'), camM[12], camM[13], camM[14]);
      gl.uniform3f(this.uloc(p, 'uSunDir'), _vSun.x, _vSun.y, _vSun.z);
      if (sun) gl.uniform3f(this.uloc(p, 'uSunColor'), sun.color.r * sun.intensity, sun.color.g * sun.intensity, sun.color.b * sun.intensity);
      else gl.uniform3f(this.uloc(p, 'uSunColor'), 2.5, 2.4, 2.2);

      if (hemi) {
        gl.uniform3f(this.uloc(p, 'uHemiSky'), hemi.color.r * hemi.intensity, hemi.color.g * hemi.intensity, hemi.color.b * hemi.intensity);
        gl.uniform3f(this.uloc(p, 'uHemiGround'), hemi.groundColor.r * hemi.intensity, hemi.groundColor.g * hemi.intensity, hemi.groundColor.b * hemi.intensity);
      } else {
        gl.uniform3f(this.uloc(p, 'uHemiSky'), 0.8, 0.9, 1.0);
        gl.uniform3f(this.uloc(p, 'uHemiGround'), 0.5, 0.48, 0.45);
      }
      gl.uniform3f(this.uloc(p, 'uSkyTop'), this.skyTop[0], this.skyTop[1], this.skyTop[2]);
      gl.uniform3f(this.uloc(p, 'uSkyHorizon'), this.skyHor[0], this.skyHor[1], this.skyHor[2]);

      if (fog) {
        gl.uniform4f(this.uloc(p, 'uFogColorNear'), fog.color.r, fog.color.g, fog.color.b, fog.near);
        gl.uniform1f(this.uloc(p, 'uFogFar'), fog.far);
      } else {
        gl.uniform4f(this.uloc(p, 'uFogColorNear'), 0.7, 0.8, 0.9, 1e6);
        gl.uniform1f(this.uloc(p, 'uFogFar'), 2e6);
      }

      gl.uniform4f(this.uloc(p, 'uSpotPosRange'), spX, spY, spZ, spRange);
      gl.uniform4f(this.uloc(p, 'uSpotDirCos'), spDx, spDy, spDz, spCos);
      gl.uniform4f(this.uloc(p, 'uSpotColorPen'), spR, spG, spB, spPenCos);
      gl.uniform1f(this.uloc(p, 'uSpotDecay'), spDecay);
      gl.uniform4f(this.uloc(p, 'uPointPosRange'), ptX, ptY, ptZ, ptRange);
      gl.uniform4f(this.uloc(p, 'uPointColorDecay'), ptR, ptG, ptB, ptDecay);

      gl.uniform1i(this.uloc(p, 'uCascadeCount'), cCount);
      gl.uniform4fv(this.uloc(p, 'uCascadeSplits'), this.cascadeSplits);
      gl.uniform4fv(this.uloc(p, 'uShadowBiases'), this.cascadeBiases);
      gl.uniform4fv(this.uloc(p, 'uShadowDepthBiases'), this.cascadeDepthBiases);
      gl.uniformMatrix4fv(this.uloc(p, 'uShadowMats[0]'), false, this.shadowMatsFlat);
    }
  }

  private computeMatFlags(mat: Material, receiveShadow: boolean, hasInstColor: boolean, toneMapInline: boolean): number {
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
    if (receiveShadow) flags |= FLAG_RECEIVE_SHADOW;
    if (mat.isMeshBasicMaterial) flags |= FLAG_UNLIT;
    if (toneMapInline) flags |= FLAG_TONEMAP_INLINE;
    if (mat.emissiveMap) flags |= FLAG_EMISSIVE_MAP;
    if (mat.alphaToCoverage && this.msaaActive) flags |= FLAG_ALPHA_TO_COV;
    return flags;
  }

  private renderMeshQueue(queue: DrawItem[], sceneEnvInt: number, toneMapInline: boolean): void {
    const gl = this.gl;
    const envBase = this.postConfig.reflections ? sceneEnvInt * 1.25 : 0;
    const exposure = this.toneMappingExposure;

    let lastMatId = -1;
    let lastFlags = -1;
    let lastProg: ProgramRec | null = null;

    for (let i = 0; i < queue.length; i++) {
      const it = queue[i];
      const mesh = it.obj;
      const geoRec = this.syncGeometry(it.geo);

      let instVao: WebGLVertexArrayObject | null = null;
      let instCount = 0;
      let hasInstColor = false;
      if (it.mode === 'instanced') {
        const im = mesh as InstancedMesh;
        const prep = this.prepareInstancedDraw(im, geoRec, _camPlanes);
        if (prep.count === 0) continue;
        instVao = prep.vao;
        instCount = prep.count;
        hasInstColor = im.instanceColor !== null;
      }

      const mat = it.mat;
      const prog = this.pbrProgs[it.mode];
      if (lastProg !== prog) {
        this.useProg(prog);
        lastProg = prog;
        lastMatId = -1;
        lastFlags = -1;
      }

      const flags = this.computeMatFlags(mat, mesh.receiveShadow, hasInstColor, toneMapInline);
      if (lastMatId !== mat.id) {
        lastMatId = mat.id;
        lastFlags = flags;
        this.setSide(mat.side);
        this.setBlend(mat.transparent, mat.blending);
        this.setDepthWrite(mat.depthWrite);
        this.setPolygonOffset(mat.polygonOffset, mat.polygonOffsetFactor, mat.polygonOffsetUnits);
        this.setAlphaToCoverage(mat.alphaToCoverage);

        gl.uniform1i(this.uloc(prog, 'uFlags'), flags);
        gl.uniform4f(this.uloc(prog, 'uBaseColor'), mat.color.r, mat.color.g, mat.color.b, mat.opacity);
        gl.uniform4f(
          this.uloc(prog, 'uEmissive'),
          mat.emissive.r * mat.emissiveIntensity,
          mat.emissive.g * mat.emissiveIntensity,
          mat.emissive.b * mat.emissiveIntensity,
          mat.alphaTest,
        );
        gl.uniform4f(this.uloc(prog, 'uPbrParams'), mat.roughness, mat.metalness, mat.clearcoat, mat.clearcoatRoughness);
        gl.uniform4f(this.uloc(prog, 'uDetailParams'), mat.bumpScale, mat.aoMapIntensity, mat.envMapIntensity * envBase, exposure);

        if (mat.map) {
          gl.uniform4f(this.uloc(prog, 'uUvTransform'), mat.map.repeat.x, mat.map.repeat.y, mat.map.offset.x, mat.map.offset.y);
          this.bindTex2D(0, this.syncTexture(mat.map, this.whiteTex));
        } else {
          gl.uniform4f(this.uloc(prog, 'uUvTransform'), 1, 1, 0, 0);
          this.bindTex2D(0, this.whiteTex);
        }
        this.bindTex2D(1, mat.emissiveMap ? this.syncTexture(mat.emissiveMap, this.blackTex) : this.blackTex);
        const crmTex = mat.roughnessMap ?? mat.metalnessMap ?? mat.clearcoatMap;
        this.bindTex2D(2, crmTex ? this.syncTexture(crmTex, this.crmDefaultTex) : this.crmDefaultTex);
        const detailTex = mat.bumpMap ?? mat.aoMap;
        this.bindTex2D(3, detailTex ? this.syncTexture(detailTex, this.whiteTex) : this.whiteTex);

        if (mat.personOutfit) {
          const po = mat.personOutfit;
          this.bindTex2D(4, this.syncTexture(po.mask, this.blackTex));
          this.bindTex2D(5, this.syncTexture(po.overlay, this.clearTex));
          gl.uniform4f(this.uloc(prog, 'uPersonTop'), po.top.x, po.top.y, po.top.z, po.top.w);
          gl.uniform4f(this.uloc(prog, 'uPersonBottom'), po.bottom.x, po.bottom.y, po.bottom.z, po.bottom.w);
          gl.uniform2f(this.uloc(prog, 'uPersonRef'), po.ref.x, po.ref.y);
        }
      } else if (lastFlags !== flags) {
        lastFlags = flags;
        gl.uniform1i(this.uloc(prog, 'uFlags'), flags);
      }

      gl.uniformMatrix4fv(this.uloc(prog, 'uModel'), false, mesh.matrixWorld.elements);

      if (it.mode === 'skinned') {
        const sk = (mesh as SkinnedMesh).skeleton;
        if (sk) {
          const count = Math.min(sk.bones.length, MAX_BONES) * 16;
          gl.uniformMatrix4fv(this.uloc(prog, 'uBones[0]'), false, sk.boneMatrices, 0, count);
        }
        this.bindVao(geoRec.vao);
        this.drawRange(geoRec, it.start, it.count, 0);
      } else if (it.mode === 'instanced') {
        this.bindVao(instVao);
        this.drawRange(geoRec, it.start, it.count, instCount);
      } else {
        this.bindVao(geoRec.vao);
        this.drawRange(geoRec, it.start, it.count, 0);
      }
    }
  }

  private drawRange(geoRec: GeoGpu, start: number, count: number, instanceCount: number): void {
    const gl = this.gl;
    if (count <= 0) return;
    if (geoRec.indexCount > 0) {
      const byteOffset = start * (geoRec.indexType === gl.UNSIGNED_INT ? 4 : 2);
      if (instanceCount > 0) {
        gl.drawElementsInstanced(gl.TRIANGLES, count, geoRec.indexType, byteOffset, instanceCount);
        this.info.render.triangles += ((count / 3) | 0) * instanceCount;
      } else {
        gl.drawElements(gl.TRIANGLES, count, geoRec.indexType, byteOffset);
        this.info.render.triangles += (count / 3) | 0;
      }
    } else {
      if (instanceCount > 0) {
        gl.drawArraysInstanced(gl.TRIANGLES, start, count, instanceCount);
        this.info.render.triangles += ((count / 3) | 0) * instanceCount;
      } else {
        gl.drawArrays(gl.TRIANGLES, start, count);
        this.info.render.triangles += (count / 3) | 0;
      }
    }
    this.info.render.calls++;
  }

  private renderLines(toneMapInline: boolean): void {
    if (this.flatLines.length === 0) return;
    const gl = this.gl;
    const p = this.linesProg;
    this.useProg(p);
    this.setDepthTest(true);
    this.setDepthWrite(false);
    this.setPolygonOffset(false);
    gl.uniformMatrix4fv(this.uloc(p, 'uViewProj'), false, _vp);
    gl.uniform1i(this.uloc(p, 'uToneMapInline'), toneMapInline ? 1 : 0);
    gl.uniform1f(this.uloc(p, 'uExposure'), this.toneMappingExposure);

    for (let i = 0; i < this.flatLines.length; i++) {
      const ls = this.flatLines[i];
      if (!isObjVisible(ls)) continue;
      const mat = ls.material;
      this.setBlend(mat.transparent, mat.blending);
      gl.uniformMatrix4fv(this.uloc(p, 'uModel'), false, ls.matrixWorld.elements);
      gl.uniform4f(this.uloc(p, 'uColor'), mat.color.r, mat.color.g, mat.color.b, mat.opacity);
      const geoRec = this.syncGeometry(ls.geometry, true);
      if (geoRec.vertexCount > 0) {
        this.bindVao(geoRec.vao);
        gl.drawArrays(gl.LINES, 0, geoRec.vertexCount);
        this.info.render.calls++;
      }
    }
    this.setDepthWrite(true);
  }

  private renderPoints(camera: Camera, toneMapInline: boolean): void {
    if (this.flatPoints.length === 0) return;
    const gl = this.gl;
    const p = this.pointsProg;
    this.useProg(p);
    this.setDepthTest(true);
    this.setDepthWrite(false);
    this.setPolygonOffset(false);
    gl.uniformMatrix4fv(this.uloc(p, 'uViewProj'), false, _vp);
    gl.uniformMatrix4fv(this.uloc(p, 'uView'), false, camera.matrixWorldInverse.elements);
    gl.uniform1f(this.uloc(p, 'uPixelScale'), this.pixelRatio);
    gl.uniform1i(this.uloc(p, 'uToneMapInline'), toneMapInline ? 1 : 0);
    gl.uniform1f(this.uloc(p, 'uExposure'), this.toneMappingExposure);

    for (let i = 0; i < this.flatPoints.length; i++) {
      const pts = this.flatPoints[i];
      if (!isObjVisible(pts)) continue;
      this.setBlend(pts.material.transparent, pts.material.blending);
      gl.uniformMatrix4fv(this.uloc(p, 'uModel'), false, pts.matrixWorld.elements);
      const geoRec = this.syncGeometry(pts.geometry, true);
      if (geoRec.vertexCount > 0) {
        this.bindVao(geoRec.vao);
        gl.drawArrays(gl.POINTS, 0, geoRec.vertexCount);
        this.info.render.calls++;
      }
    }
    this.setDepthWrite(true);
  }

  private renderPostProcess(camera: Camera): void {
    const gl = this.gl;
    this.setDepthTest(false);
    this.setDepthWrite(false);
    this.setBlend(false, 0);
    this.setSide(DoubleSide);
    this.setPolygonOffset(false);
    this.setAlphaToCoverage(false);
    this.bindVao(this.emptyVao);

    const hasAo = Boolean(this.aoFbo && this.aoTex && this.postAoW > 0);
    if (hasAo) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.aoFbo);
      gl.viewport(0, 0, this.postAoW, this.postAoH);
      const p = this.ssaoProg;
      this.useProg(p);
      this.bindTex2D(0, this.sceneDepthTex!);
      const pCam = camera as PerspectiveCamera;
      const tanH = Math.tan(((pCam.fov || 62) * Math.PI) / 360);
      const aspect = pCam.aspect || (this.fbW / Math.max(1, this.fbH));
      gl.uniform2f(this.uloc(p, 'uFullTexel'), 1 / this.fbW, 1 / this.fbH);
      gl.uniform4f(this.uloc(p, 'uProjParams'), camera.near, camera.far, tanH * aspect, tanH);
      gl.uniform1i(this.uloc(p, 'uSamples'), this.postConfig.aoScale < 0.7 ? 6 : 8);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      this.info.render.calls++;
    }

    const bw = Math.max(1, this.fbW >> 2), bh = Math.max(1, this.fbH >> 2);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.bloomFboA);
    gl.viewport(0, 0, bw, bh);
    this.useProg(this.bloomExtractProg);
    this.bindTex2D(0, this.sceneColorTex!);
    gl.uniform2f(this.uloc(this.bloomExtractProg, 'uTexel'), 1 / this.fbW, 1 / this.fbH);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    this.info.render.calls++;

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.bloomFboB);
    this.useProg(this.bloomBlurProg);
    this.bindTex2D(0, this.bloomTexA!);
    gl.uniform2f(this.uloc(this.bloomBlurProg, 'uDir'), 1.5 / bw, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    this.info.render.calls++;

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.bloomFboA);
    this.bindTex2D(0, this.bloomTexB!);
    gl.uniform2f(this.uloc(this.bloomBlurProg, 'uDir'), 0, 1.5 / bh);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    this.info.render.calls++;

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.fbW, this.fbH);
    const cp = this.compositeProg;
    this.useProg(cp);
    this.bindTex2D(0, this.sceneColorTex!);
    this.bindTex2D(1, this.bloomTexA!);
    this.bindTex2D(2, hasAo ? this.aoTex! : this.whiteTex);
    gl.uniform2f(this.uloc(cp, 'uAoTexel'), hasAo ? 1 / this.postAoW : 0, hasAo ? 1 / this.postAoH : 0);
    gl.uniform1f(this.uloc(cp, 'uBloomStrength'), this.postConfig.bloomStrength);
    gl.uniform1f(this.uloc(cp, 'uAoEnabled'), hasAo ? 1 : 0);
    gl.uniform1f(this.uloc(cp, 'uExposure'), this.toneMappingExposure);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    this.info.render.calls++;
  }

  dispose(): void {
    const gl = this.gl;
    this.freePostTargets();
    if (this.shadowFbo) { gl.deleteFramebuffer(this.shadowFbo); this.shadowFbo = null; }
    if (this.shadowArrayTex) { gl.deleteTexture(this.shadowArrayTex); this.shadowArrayTex = null; }
    for (const r of this.geoCache.values()) this.deleteGeoGpu(r);
    this.geoCache.clear();
    for (const t of this.texCache.values()) gl.deleteTexture(t.glTex);
    this.texCache.clear();
    for (const e of this.instBufCache.values()) {
      gl.deleteVertexArray(e.vao);
      gl.deleteBuffer(e.matBuf);
      if (e.colBuf) gl.deleteBuffer(e.colBuf);
    }
    this.instBufCache.clear();
    for (const p of this.info.programs) gl.deleteProgram(p.program);
    this.info.programs.length = 0;
    gl.deleteVertexArray(this.emptyVao);
    gl.deleteTexture(this.whiteTex);
    gl.deleteTexture(this.blackTex);
    gl.deleteTexture(this.crmDefaultTex);
    gl.deleteTexture(this.clearTex);
    gl.deleteTexture(this.dummyShadowArray);
  }
}
