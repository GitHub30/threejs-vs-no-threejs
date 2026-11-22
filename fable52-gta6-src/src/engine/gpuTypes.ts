// Zero-dependency WebGPU type definitions & unified EngineRenderer contract.
// Avoids adding @webgpu/types while keeping full strict TypeScript safety.

import type { Camera, Scene } from './scene';

export type RendererBackend = 'webgpu' | 'webgl2';

export interface PostConfig {
  enabled: boolean;
  aoScale: number;        // 0 = off, 0.5 = half-res, 0.75..1 = high-res
  bloomStrength: number;
  reflections: boolean;
}

export interface ShadowConfig {
  enabled: boolean;
  mapSize: number;
  cascades: number[] | null;
}

export interface RendererOptions {
  canvas: HTMLCanvasElement;
  antialias?: boolean;
  powerPreference?: WebGLPowerPreference;
  preferredBackend?: 'auto' | RendererBackend;
}

export interface EngineRenderer {
  readonly backend: RendererBackend;
  readonly domElement: HTMLCanvasElement;
  readonly capabilities: { maxTextureSize: number; maxAnisotropy: number; maxSamples: number };
  shadowMap: { enabled: boolean; type: number };
  toneMapping: number;
  toneMappingExposure: number;
  postConfig: PostConfig;
  shadowConfig: ShadowConfig;
  info: {
    render: { calls: number; triangles: number };
    memory: { geometries: number; textures: number };
    programs: unknown[];
  };
  getContext(): WebGL2RenderingContext | GPUCanvasContextLike;
  setPixelRatio(pr: number): void;
  setSize(w: number, h: number, updateStyle?: boolean): void;
  render(scene: Scene, camera: Camera): void;
  dispose(): void;
}

// WebGPU Usage Bitflags (spec constants)
export const GPU_BUFFER_USAGE = {
  MAP_READ: 0x0001,
  MAP_WRITE: 0x0002,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  INDEX: 0x0010,
  VERTEX: 0x0020,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
  INDIRECT: 0x0100,
} as const;

export const GPU_TEXTURE_USAGE = {
  COPY_SRC: 0x01,
  COPY_DST: 0x02,
  TEXTURE_BINDING: 0x04,
  STORAGE_BINDING: 0x08,
  RENDER_ATTACHMENT: 0x10,
} as const;

export const GPU_SHADER_STAGE = {
  VERTEX: 0x1,
  FRAGMENT: 0x2,
  COMPUTE: 0x4,
} as const;

// Minimal structural interfaces for WebGPU API (compatible with Chrome, Safari, Firefox)
export interface GPUNavigatorLike {
  requestAdapter(opts?: { powerPreference?: string; forceFallbackAdapter?: boolean }): Promise<GPUAdapterLike | null>;
  getPreferredCanvasFormat(): string;
}

export interface GPUAdapterLike {
  readonly features?: Set<string>;
  readonly limits: Record<string, number>;
  readonly info?: { vendor?: string; architecture?: string; device?: string; description?: string; isFallbackAdapter?: boolean };
  requestDevice(desc?: { requiredLimits?: Record<string, number> }): Promise<GPUDeviceLike>;
}

export interface GPUDeviceLike {
  readonly limits: Record<string, number>;
  readonly queue: GPUQueueLike;
  readonly lost: Promise<{ reason?: string; message?: string }>;
  createBuffer(desc: { size: number; usage: number; mappedAtCreation?: boolean }): GPUBufferLike;
  createTexture(desc: {
    size: [number, number] | [number, number, number];
    mipLevelCount?: number;
    sampleCount?: number;
    dimension?: '2d';
    format: string;
    usage: number;
    viewFormats?: string[];
  }): GPUTextureLike;
  createSampler(desc?: {
    addressModeU?: string;
    addressModeV?: string;
    magFilter?: string;
    minFilter?: string;
    mipmapFilter?: string;
    compare?: string;
    maxAnisotropy?: number;
  }): GPUSamplerLike;
  createShaderModule(desc: { code: string }): GPUShaderModuleLike;
  createBindGroupLayout(desc: { entries: unknown[] }): GPUBindGroupLayoutLike;
  createPipelineLayout(desc: { bindGroupLayouts: GPUBindGroupLayoutLike[] }): GPUPipelineLayoutLike;
  createBindGroup(desc: { layout: GPUBindGroupLayoutLike; entries: unknown[] }): GPUBindGroupLike;
  createRenderPipeline(desc: unknown): GPURenderPipelineLike;
  createRenderPipelineAsync(desc: unknown): Promise<GPURenderPipelineLike>;
  createComputePipeline(desc: unknown): GPUComputePipelineLike;
  createComputePipelineAsync(desc: unknown): Promise<GPUComputePipelineLike>;
  createCommandEncoder(): GPUCommandEncoderLike;
  pushErrorScope(filter: 'validation' | 'out-of-memory' | 'internal'): void;
  popErrorScope(): Promise<{ message: string } | null>;
  destroy(): void;
}

export interface GPUQueueLike {
  submit(cmds: GPUCommandBufferLike[]): void;
  writeBuffer(buf: GPUBufferLike, bufferOffset: number, data: ArrayBuffer | ArrayBufferView, dataOffset?: number, size?: number): void;
  writeTexture(
    dest: { texture: GPUTextureLike; mipLevel?: number; origin?: [number, number, number] },
    data: ArrayBuffer | ArrayBufferView,
    dataLayout: { offset?: number; bytesPerRow: number; rowsPerImage?: number },
    size: [number, number] | [number, number, number],
  ): void;
  copyExternalImageToTexture(
    src: { source: TexImageSource; flipY?: boolean },
    dst: { texture: GPUTextureLike; mipLevel?: number; premultipliedAlpha?: boolean },
    size: [number, number],
  ): void;
  onSubmittedWorkDone(): Promise<void>;
}

export interface GPUCanvasContextLike {
  configure(cfg: { device: GPUDeviceLike; format: string; alphaMode?: 'opaque' | 'premultiplied'; usage?: number }): void;
  unconfigure(): void;
  getCurrentTexture(): GPUTextureLike;
}

export interface GPUBufferLike {
  readonly size: number;
  destroy(): void;
}

export interface GPUTextureLike {
  readonly width: number;
  readonly height: number;
  createView(desc?: {
    format?: string;
    dimension?: '2d' | '2d-array';
    aspect?: 'all' | 'depth-only';
    baseMipLevel?: number;
    mipLevelCount?: number;
    baseArrayLayer?: number;
    arrayLayerCount?: number;
  }): GPUTextureViewLike;
  destroy(): void;
}

export interface GPUTextureViewLike {}
export interface GPUSamplerLike {}
export interface GPUShaderModuleLike {
  getCompilationInfo?(): Promise<{ messages: { type: string; message: string; lineNum: number; linePos: number }[] }>;
}
export interface GPUBindGroupLayoutLike {}
export interface GPUPipelineLayoutLike {}
export interface GPUBindGroupLike {}
export interface GPURenderPipelineLike {}
export interface GPUComputePipelineLike {}
export interface GPUCommandBufferLike {}

export interface GPUCommandEncoderLike {
  beginRenderPass(desc: unknown): GPURenderPassEncoderLike;
  beginComputePass(): GPUComputePassEncoderLike;
  copyBufferToBuffer(src: GPUBufferLike, srcOff: number, dst: GPUBufferLike, dstOff: number, size: number): void;
  finish(): GPUCommandBufferLike;
}

export interface GPURenderPassEncoderLike {
  setPipeline(p: GPURenderPipelineLike): void;
  setBindGroup(index: number, bg: GPUBindGroupLike, dynamicOffsets?: number[]): void;
  setVertexBuffer(slot: number, buf: GPUBufferLike, offset?: number, size?: number): void;
  setIndexBuffer(buf: GPUBufferLike, format: 'uint16' | 'uint32', offset?: number, size?: number): void;
  setViewport(x: number, y: number, w: number, h: number, minD: number, maxD: number): void;
  draw(vertexCount: number, instanceCount?: number, firstVertex?: number, firstInstance?: number): void;
  drawIndexed(indexCount: number, instanceCount?: number, firstIndex?: number, baseVertex?: number, firstInstance?: number): void;
  drawIndirect(indirectBuffer: GPUBufferLike, indirectOffset: number): void;
  drawIndexedIndirect(indirectBuffer: GPUBufferLike, indirectOffset: number): void;
  end(): void;
}

export interface GPUComputePassEncoderLike {
  setPipeline(p: GPUComputePipelineLike): void;
  setBindGroup(index: number, bg: GPUBindGroupLike, dynamicOffsets?: number[]): void;
  dispatchWorkgroups(x: number, y?: number, z?: number): void;
  end(): void;
}
