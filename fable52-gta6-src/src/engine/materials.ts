import {
  ClampToEdgeWrapping,
  Color,
  ColorLike,
  FrontSide,
  NoColorSpace,
  NormalBlending,
  Vector2,
  Vector4,
} from './math';

let nextTexId = 1;
let nextMatId = 1;

export type TextureImage = HTMLImageElement | HTMLCanvasElement | ImageBitmap | OffscreenCanvas | null;

export class Texture {
  readonly id = nextTexId++;
  image: TextureImage;
  wrapS = ClampToEdgeWrapping;
  wrapT = ClampToEdgeWrapping;
  repeat = new Vector2(1, 1);
  offset = new Vector2(0, 0);
  anisotropy = 1;
  colorSpace: string = NoColorSpace;
  channel = 0;
  flipY = true;
  version = 1;
  _disposedVersion = 0;
  _onDispose: ((t: Texture) => void) | null = null;
  isDataTexture = false;
  isCanvasTexture = false;
  data: Uint8Array | null = null;
  width = 0;
  height = 0;

  constructor(image: TextureImage = null) {
    this.image = image;
    if (image) {
      this.width = (image as { width: number }).width || 0;
      this.height = (image as { height: number }).height || 0;
    }
  }

  set needsUpdate(v: boolean) {
    if (v) this.version++;
  }

  dispose(): void {
    this._disposedVersion++;
    this._onDispose?.(this);
  }
}

export class CanvasTexture extends Texture {
  constructor(canvas: HTMLCanvasElement) {
    super(canvas);
    this.isCanvasTexture = true;
    this.version = 1;
  }
}

export class DataTexture extends Texture {
  constructor(data: Uint8Array, width: number, height: number) {
    super(null);
    this.isDataTexture = true;
    this.data = data;
    this.width = width;
    this.height = height;
    this.flipY = false;
    this.version = 1;
  }
}

export class TextureLoader {
  load(
    url: string,
    onLoad?: (t: Texture) => void,
    _onProgress?: unknown,
    onError?: (err: unknown) => void,
  ): Texture {
    const tex = new Texture(null);
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      tex.image = img;
      tex.width = img.naturalWidth || img.width;
      tex.height = img.naturalHeight || img.height;
      tex.version++;
      onLoad?.(tex);
    };
    img.onerror = (err) => onError?.(err);
    img.src = url;
    return tex;
  }
}

export interface PersonOutfitUniforms {
  mask: Texture;
  overlay: Texture;
  top: Vector4;
  bottom: Vector4;
  ref: Vector2;
}

export interface MaterialParams {
  color?: ColorLike;
  map?: Texture | null;
  emissive?: ColorLike;
  emissiveMap?: Texture | null;
  emissiveIntensity?: number;
  roughness?: number;
  metalness?: number;
  roughnessMap?: Texture | null;
  metalnessMap?: Texture | null;
  clearcoat?: number;
  clearcoatMap?: Texture | null;
  clearcoatRoughness?: number;
  bumpMap?: Texture | null;
  bumpScale?: number;
  aoMap?: Texture | null;
  aoMapIntensity?: number;
  envMapIntensity?: number;
  vertexColors?: boolean;
  flatShading?: boolean;
  transparent?: boolean;
  opacity?: number;
  alphaTest?: number;
  alphaToCoverage?: boolean;
  side?: number;
  depthWrite?: boolean;
  blending?: number;
  fog?: boolean;
  polygonOffset?: boolean;
  polygonOffsetFactor?: number;
  polygonOffsetUnits?: number;
  visible?: boolean;
  carPaintShell?: boolean;
  personOutfit?: PersonOutfitUniforms | null;
}

export class Material {
  readonly id = nextMatId++;
  readonly kind: 'standard' | 'physical' | 'basic' | 'line' | 'shader' = 'standard';
  color = new Color(0xffffff);
  map: Texture | null = null;
  emissive = new Color(0x000000);
  emissiveMap: Texture | null = null;
  emissiveIntensity = 1;
  roughness = 0.85;
  metalness = 0;
  roughnessMap: Texture | null = null;
  metalnessMap: Texture | null = null;
  clearcoat = 0;
  clearcoatMap: Texture | null = null;
  clearcoatRoughness = 0.08;
  bumpMap: Texture | null = null;
  bumpScale = 1;
  aoMap: Texture | null = null;
  aoMapIntensity = 1;
  envMapIntensity = 1;
  vertexColors = false;
  flatShading = false;
  transparent = false;
  opacity = 1;
  alphaTest = 0;
  alphaToCoverage = false;
  side = FrontSide;
  depthWrite = true;
  blending = NormalBlending;
  fog = true;
  polygonOffset = false;
  polygonOffsetFactor = 0;
  polygonOffsetUnits = 0;
  visible = true;
  carPaintShell = false;
  personOutfit: PersonOutfitUniforms | null = null;
  version = 1;

  isMeshStandardMaterial = false;
  isMeshPhysicalMaterial = false;
  isMeshBasicMaterial = false;
  isLineBasicMaterial = false;
  isShaderMaterial = false;

  constructor(params?: MaterialParams) {
    if (!params) return;
    if (params.color !== undefined) this.color.set(params.color);
    if (params.map !== undefined) this.map = params.map;
    if (params.emissive !== undefined) this.emissive.set(params.emissive);
    if (params.emissiveMap !== undefined) this.emissiveMap = params.emissiveMap;
    if (params.emissiveIntensity !== undefined) this.emissiveIntensity = params.emissiveIntensity;
    if (params.roughness !== undefined) this.roughness = params.roughness;
    if (params.metalness !== undefined) this.metalness = params.metalness;
    if (params.roughnessMap !== undefined) this.roughnessMap = params.roughnessMap;
    if (params.metalnessMap !== undefined) this.metalnessMap = params.metalnessMap;
    if (params.clearcoat !== undefined) this.clearcoat = params.clearcoat;
    if (params.clearcoatMap !== undefined) this.clearcoatMap = params.clearcoatMap;
    if (params.clearcoatRoughness !== undefined) this.clearcoatRoughness = params.clearcoatRoughness;
    if (params.bumpMap !== undefined) this.bumpMap = params.bumpMap;
    if (params.bumpScale !== undefined) this.bumpScale = params.bumpScale;
    if (params.aoMap !== undefined) this.aoMap = params.aoMap;
    if (params.aoMapIntensity !== undefined) this.aoMapIntensity = params.aoMapIntensity;
    if (params.envMapIntensity !== undefined) this.envMapIntensity = params.envMapIntensity;
    if (params.vertexColors !== undefined) this.vertexColors = params.vertexColors;
    if (params.flatShading !== undefined) this.flatShading = params.flatShading;
    if (params.transparent !== undefined) this.transparent = params.transparent;
    if (params.opacity !== undefined) this.opacity = params.opacity;
    if (params.alphaTest !== undefined) this.alphaTest = params.alphaTest;
    if (params.alphaToCoverage !== undefined) this.alphaToCoverage = params.alphaToCoverage;
    if (params.side !== undefined) this.side = params.side;
    if (params.depthWrite !== undefined) this.depthWrite = params.depthWrite;
    if (params.blending !== undefined) this.blending = params.blending;
    if (params.fog !== undefined) this.fog = params.fog;
    if (params.polygonOffset !== undefined) this.polygonOffset = params.polygonOffset;
    if (params.polygonOffsetFactor !== undefined) this.polygonOffsetFactor = params.polygonOffsetFactor;
    if (params.polygonOffsetUnits !== undefined) this.polygonOffsetUnits = params.polygonOffsetUnits;
    if (params.visible !== undefined) this.visible = params.visible;
    if (params.carPaintShell !== undefined) this.carPaintShell = params.carPaintShell;
    if (params.personOutfit !== undefined) this.personOutfit = params.personOutfit;
  }

  set needsUpdate(v: boolean) {
    if (v) this.version++;
  }

  dispose(): void {
    this.version++;
  }
}

export class MeshStandardMaterial extends Material {
  override readonly kind = 'standard' as const;
  constructor(params?: MaterialParams) {
    super(params);
    this.isMeshStandardMaterial = true;
  }
}

export class MeshPhysicalMaterial extends Material {
  override readonly kind = 'physical' as const;
  constructor(params?: MaterialParams) {
    super(params);
    this.isMeshStandardMaterial = true;
    this.isMeshPhysicalMaterial = true;
  }
}

export class MeshBasicMaterial extends Material {
  override readonly kind = 'basic' as const;
  constructor(params?: MaterialParams) {
    super(params);
    this.isMeshBasicMaterial = true;
  }
}

export class LineBasicMaterial extends Material {
  override readonly kind = 'line' as const;
  constructor(params?: MaterialParams) {
    super(params);
    this.isLineBasicMaterial = true;
  }
}

export interface ShaderMaterialParams extends MaterialParams {
  uniforms?: Record<string, { value: unknown }>;
  vertexShader?: string;
  fragmentShader?: string;
}

export class ShaderMaterial extends Material {
  override readonly kind = 'shader' as const;
  uniforms: Record<string, { value: unknown }>;
  vertexShader: string;
  fragmentShader: string;

  constructor(params?: ShaderMaterialParams) {
    super(params);
    this.isShaderMaterial = true;
    this.uniforms = params?.uniforms ?? {};
    this.vertexShader = params?.vertexShader ?? '';
    this.fragmentShader = params?.fragmentShader ?? '';
  }
}
