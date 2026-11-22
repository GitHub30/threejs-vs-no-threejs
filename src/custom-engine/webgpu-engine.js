// Bespoke zero-dependency Custom Game Engine powered by Rust Wasm SIMD128 + Raw WebGPU
// Features: GPURenderBundle pre-recorded passes, cooperative workgroup shared-memory compute,
// 32-byte compact instance pulling, 4x MSAA, zero-GC hot loop, and hardware GPU timestamp queries.

import { BOID_PARAMS, COMPUTE_PEER_BUDGET, SUN_DIR } from '../shared/config.js';
import {
  createArenaGeometryData,
  createBirdGeometryData,
  createCrystalGeometryData,
} from '../shared/geometry.js';
import { generateInitialBoids, generateInitialCrystals } from '../shared/seed.js';
import { loadWasmCore } from './wasm-bridge.js';
import {
  ARENA_SHADER_WGSL,
  BIRD_RENDER_WGSL,
  BOIDS_COMPUTE_WGSL,
  CRYSTAL_RENDER_WGSL,
  FLOOR_SHADER_WGSL,
  SKY_SHADER_WGSL,
} from './shaders.js';

const MAX_INSTANCES = 65536;
const INSTANCE_STRIDE_BYTES = 32; // 8 x f32

export class CustomWebGPUEngine {
  constructor(canvas) {
    this.canvas = canvas;
    this.device = null;
    this.context = null;
    this.format = null;
    this.wasm = null;
    this.msaaSamples = 4;

    this.scenario = 'hybrid';
    this.entityCount = 8192;
    this.initialized = false;
    this.disposed = false;

    this.initMs = 0;
    this.backendLabel = 'WebGPU';
    this.wasmLabel = '';
    this.hasTimestampQuery = false;
    this.lastGpuMs = -1;

    this._width = 0;
    this._height = 0;
    this._msaaTex = null;
    this._depthTex = null;

    // Pre-allocated uniform typed arrays (zero per-frame allocations)
    this._frameData = new Float32Array(40); // 2*16 + 4 + 4 = 40 floats = 160 bytes
    this._simDataBuf = new ArrayBuffer(48);
    this._simU32 = new Uint32Array(this._simDataBuf);
    this._simI32 = new Int32Array(this._simDataBuf);
    this._simF32 = new Float32Array(this._simDataBuf);

    this._computeStep = 0;
    this._pingPong = 0;
    this._time = 0;

    // GPU timestamp double-buffered async readback
    this._tsQuerySet = null;
    this._tsResolveBuf = null;
    this._tsReadBufs = [];
    this._tsSlot = 0;
    this._tsPending = [false, false];

    // Pre-recorded RenderBundles
    this._envBundle = null;
    this._computeBundles = [null, null];
    this._crystalBundle = null;
    this._crystalBundleCount = 0;
  }

  static async isAvailable() {
    if (!navigator.gpu) return false;
    try {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      return !!adapter;
    } catch {
      return false;
    }
  }

  async init(scenario = 'hybrid', count = 8192) {
    const t0 = performance.now();
    this.scenario = scenario;
    this.entityCount = count;

    const [wasm, adapter] = await Promise.all([
      loadWasmCore(),
      navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }),
    ]);

    if (!adapter) {
      throw new Error('WebGPU adapter unavailable');
    }
    this.wasm = wasm;
    this.wasmLabel = wasm.label;

    const requiredFeatures = [];
    if (adapter.features.has('timestamp-query')) {
      requiredFeatures.push('timestamp-query');
      this.hasTimestampQuery = true;
    }

    this.device = await adapter.requestDevice({ requiredFeatures });
    this.device.lost.then(() => {
      this.initialized = false;
    });

    const adapterDesc = adapter.info?.architecture || adapter.info?.vendor || '';
    this.backendLabel = adapterDesc
      ? `WebGPU (${adapterDesc})`
      : 'WebGPU';

    this.context = this.canvas.getContext('webgpu');
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({
      device: this.device,
      format: this.format,
      alphaMode: 'opaque',
    });

    this._createUniformBuffers();
    this._createGeometryBuffers();
    this._createPipelines();
    this._createBindGroups();
    if (this.hasTimestampQuery) {
      this._createTimestampResources();
    }
    this._buildEnvBundle();

    this.resize(this.canvas.clientWidth || 640, this.canvas.clientHeight || 480);
    this.configure(scenario, count);

    this.initMs = performance.now() - t0;
    this.initialized = true;
  }

  _createUniformBuffers() {
    const d = this.device;
    this.frameUniformBuf = d.createBuffer({
      size: 160,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.simUniformBuf = d.createBuffer({
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // Storage buffers for dynamic instances (Hybrid / Crystals) and Ping-Pong Compute
    const bufBytes = MAX_INSTANCES * INSTANCE_STRIDE_BYTES;
    this.dynamicInstanceBuf = d.createBuffer({
      size: bufBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.computeBufA = d.createBuffer({
      size: bufBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.computeBufB = d.createBuffer({
      size: bufBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }

  _createGeometryBuffers() {
    const d = this.device;

    // Static Arena Monoliths (interleaved pos3 + norm3 + col3 = 36B)
    const arena = createArenaGeometryData();
    this.arenaVertCount = arena.vertexCount;
    const arenaData = new Float32Array(arena.vertexCount * 9);
    for (let i = 0; i < arena.vertexCount; i++) {
      arenaData[i * 9]     = arena.positions[i * 3];
      arenaData[i * 9 + 1] = arena.positions[i * 3 + 1];
      arenaData[i * 9 + 2] = arena.positions[i * 3 + 2];
      arenaData[i * 9 + 3] = arena.normals[i * 3];
      arenaData[i * 9 + 4] = arena.normals[i * 3 + 1];
      arenaData[i * 9 + 5] = arena.normals[i * 3 + 2];
      arenaData[i * 9 + 6] = arena.colors[i * 3];
      arenaData[i * 9 + 7] = arena.colors[i * 3 + 1];
      arenaData[i * 9 + 8] = arena.colors[i * 3 + 2];
    }
    this.arenaVB = this._uploadStaticBuffer(arenaData, GPUBufferUsage.VERTEX);

    // Sculpted Origami Bird (interleaved pos3 + norm3 + wingExtra2 = 32B)
    const bird = createBirdGeometryData();
    this.birdVertCount = bird.vertexCount;
    const birdData = new Float32Array(bird.vertexCount * 8);
    for (let i = 0; i < bird.vertexCount; i++) {
      birdData[i * 8]     = bird.positions[i * 3];
      birdData[i * 8 + 1] = bird.positions[i * 3 + 1];
      birdData[i * 8 + 2] = bird.positions[i * 3 + 2];
      birdData[i * 8 + 3] = bird.normals[i * 3];
      birdData[i * 8 + 4] = bird.normals[i * 3 + 1];
      birdData[i * 8 + 5] = bird.normals[i * 3 + 2];
      birdData[i * 8 + 6] = bird.wingExtra[i * 2];
      birdData[i * 8 + 7] = bird.wingExtra[i * 2 + 1];
    }
    this.birdVB = this._uploadStaticBuffer(birdData, GPUBufferUsage.VERTEX);

    // Faceted Crystal Beacon (interleaved pos3 + norm3 = 24B)
    const crystal = createCrystalGeometryData();
    this.crystalVertCount = crystal.vertexCount;
    const crystalData = new Float32Array(crystal.vertexCount * 6);
    for (let i = 0; i < crystal.vertexCount; i++) {
      crystalData[i * 6]     = crystal.positions[i * 3];
      crystalData[i * 6 + 1] = crystal.positions[i * 3 + 1];
      crystalData[i * 6 + 2] = crystal.positions[i * 3 + 2];
      crystalData[i * 6 + 3] = crystal.normals[i * 3];
      crystalData[i * 6 + 4] = crystal.normals[i * 3 + 1];
      crystalData[i * 6 + 5] = crystal.normals[i * 3 + 2];
    }
    this.crystalVB = this._uploadStaticBuffer(crystalData, GPUBufferUsage.VERTEX);
  }

  _uploadStaticBuffer(typedArray, usage) {
    const buf = this.device.createBuffer({
      size: (typedArray.byteLength + 3) & ~3,
      usage: usage | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(buf, 0, typedArray);
    return buf;
  }

  _createPipelines() {
    const d = this.device;
    const targets = [{ format: this.format }];
    const multisample = { count: this.msaaSamples };

    // Shared bind group layouts
    this.envBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });
    this.instBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });
    this.computeBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });

    const envLayout = d.createPipelineLayout({ bindGroupLayouts: [this.envBGL] });
    const instLayout = d.createPipelineLayout({ bindGroupLayouts: [this.instBGL] });
    const compLayout = d.createPipelineLayout({ bindGroupLayouts: [this.computeBGL] });

    // 1. Sky Dome Pipeline (drawn LAST with depthCompare: 'less-equal' at z=1.0 for hardware Early-Z rejection!)
    const skyMod = d.createShaderModule({ code: SKY_SHADER_WGSL });
    this.skyPipeline = d.createRenderPipeline({
      layout: envLayout,
      vertex: { module: skyMod, entryPoint: 'vs_main' },
      fragment: { module: skyMod, entryPoint: 'fs_main', targets },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less-equal' },
      multisample,
    });

    // 2. Floor Pipeline
    const floorMod = d.createShaderModule({ code: FLOOR_SHADER_WGSL });
    this.floorPipeline = d.createRenderPipeline({
      layout: envLayout,
      vertex: { module: floorMod, entryPoint: 'vs_main' },
      fragment: { module: floorMod, entryPoint: 'fs_main', targets },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
      multisample,
    });

    // 3. Monolith Arena Pipeline
    const arenaMod = d.createShaderModule({ code: ARENA_SHADER_WGSL });
    this.arenaPipeline = d.createRenderPipeline({
      layout: envLayout,
      vertex: {
        module: arenaMod,
        entryPoint: 'vs_main',
        buffers: [{
          arrayStride: 36,
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' },
            { shaderLocation: 1, offset: 12, format: 'float32x3' },
            { shaderLocation: 2, offset: 24, format: 'float32x3' },
          ],
        }],
      },
      fragment: { module: arenaMod, entryPoint: 'fs_main', targets },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
      multisample,
    });

    // 4. Articulated Origami Bird Pipeline (shared by Scenario 1 & Scenario 2)
    const birdMod = d.createShaderModule({ code: BIRD_RENDER_WGSL });
    this.birdPipeline = d.createRenderPipeline({
      layout: instLayout,
      vertex: {
        module: birdMod,
        entryPoint: 'vs_main',
        buffers: [{
          arrayStride: 32,
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' },
            { shaderLocation: 1, offset: 12, format: 'float32x3' },
            { shaderLocation: 2, offset: 24, format: 'float32x2' },
          ],
        }],
      },
      fragment: { module: birdMod, entryPoint: 'fs_main', targets },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
      multisample,
    });

    // 5. Pure GPU Compute Boids Pipeline (Scenario 2)
    const compMod = d.createShaderModule({ code: BOIDS_COMPUTE_WGSL });
    this.computePipeline = d.createComputePipeline({
      layout: compLayout,
      compute: { module: compMod, entryPoint: 'cs_main' },
    });

    // 6. Crystal Beacons Pipeline (Scenario 3 - Multi-Draw GPURenderBundle)
    const crystalMod = d.createShaderModule({ code: CRYSTAL_RENDER_WGSL });
    this.crystalPipeline = d.createRenderPipeline({
      layout: instLayout,
      vertex: {
        module: crystalMod,
        entryPoint: 'vs_main',
        buffers: [{
          arrayStride: 24,
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' },
            { shaderLocation: 1, offset: 12, format: 'float32x3' },
          ],
        }],
      },
      fragment: { module: crystalMod, entryPoint: 'fs_main', targets },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
      multisample,
    });
  }

  _createBindGroups() {
    const d = this.device;
    this.envBG = d.createBindGroup({
      layout: this.envBGL,
      entries: [{ binding: 0, resource: { buffer: this.frameUniformBuf } }],
    });

    this.dynamicInstBG = d.createBindGroup({
      layout: this.instBGL,
      entries: [
        { binding: 0, resource: { buffer: this.frameUniformBuf } },
        { binding: 1, resource: { buffer: this.dynamicInstanceBuf } },
      ],
    });

    this.computeRenderBGs = [
      d.createBindGroup({
        layout: this.instBGL,
        entries: [
          { binding: 0, resource: { buffer: this.frameUniformBuf } },
          { binding: 1, resource: { buffer: this.computeBufA } },
        ],
      }),
      d.createBindGroup({
        layout: this.instBGL,
        entries: [
          { binding: 0, resource: { buffer: this.frameUniformBuf } },
          { binding: 1, resource: { buffer: this.computeBufB } },
        ],
      }),
    ];

    this.computeBGs = [
      d.createBindGroup({
        layout: this.computeBGL,
        entries: [
          { binding: 0, resource: { buffer: this.simUniformBuf } },
          { binding: 1, resource: { buffer: this.computeBufA } },
          { binding: 2, resource: { buffer: this.computeBufB } },
        ],
      }),
      d.createBindGroup({
        layout: this.computeBGL,
        entries: [
          { binding: 0, resource: { buffer: this.simUniformBuf } },
          { binding: 1, resource: { buffer: this.computeBufB } },
          { binding: 2, resource: { buffer: this.computeBufA } },
        ],
      }),
    ];
  }

  _createTimestampResources() {
    const d = this.device;
    this._tsQuerySet = d.createQuerySet({ type: 'timestamp', count: 4 });
    this._tsResolveBuf = d.createBuffer({
      size: 32,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    this._tsReadBufs = [
      d.createBuffer({ size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
      d.createBuffer({ size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
    ];
  }

  _buildEnvBundle() {
    // Opaque monoliths drawn FIRST (maximizes Early-Z rejection for entities, floor, and sky)
    const encArena = this.device.createRenderBundleEncoder({
      colorFormats: [this.format],
      depthStencilFormat: 'depth24plus',
      sampleCount: this.msaaSamples,
    });
    encArena.setPipeline(this.arenaPipeline);
    encArena.setBindGroup(0, this.envBG);
    encArena.setVertexBuffer(0, this.arenaVB);
    encArena.draw(this.arenaVertCount);
    this._arenaBundle = encArena.finish();

    // Floor + Sky drawn AFTER opaque foreground entities for maximal hardware Early-Z culling
    const encBackdrop = this.device.createRenderBundleEncoder({
      colorFormats: [this.format],
      depthStencilFormat: 'depth24plus',
      sampleCount: this.msaaSamples,
    });
    encBackdrop.setPipeline(this.floorPipeline);
    encBackdrop.setBindGroup(0, this.envBG);
    encBackdrop.draw(6);
    encBackdrop.setPipeline(this.skyPipeline);
    encBackdrop.setBindGroup(0, this.envBG);
    encBackdrop.draw(3);
    this._backdropBundle = encBackdrop.finish();
  }

  _buildComputeBundles(count) {
    for (let i = 0; i < 2; i++) {
      const enc = this.device.createRenderBundleEncoder({
        colorFormats: [this.format],
        depthStencilFormat: 'depth24plus',
        sampleCount: this.msaaSamples,
      });
      enc.setPipeline(this.birdPipeline);
      enc.setBindGroup(0, this.computeRenderBGs[i]);
      enc.setVertexBuffer(0, this.birdVB);
      enc.draw(this.birdVertCount, count, 0, 0);
      this._computeBundles[i] = enc.finish();
    }
  }

  _buildCrystalBundle(count) {
    if (this._crystalBundleCount === count && this._crystalBundle) return;
    const enc = this.device.createRenderBundleEncoder({
      colorFormats: [this.format],
      depthStencilFormat: 'depth24plus',
      sampleCount: this.msaaSamples,
    });
    enc.setPipeline(this.crystalPipeline);
    enc.setBindGroup(0, this.dynamicInstBG);
    enc.setVertexBuffer(0, this.crystalVB);
    // Genuine N individual draw calls pre-recorded in a WebGPU GPURenderBundle!
    for (let i = 0; i < count; i++) {
      enc.draw(this.crystalVertCount, 1, 0, i);
    }
    this._crystalBundle = enc.finish();
    this._crystalBundleCount = count;
  }

  configure(scenario, count) {
    this.scenario = scenario;
    this.entityCount = Math.min(count, MAX_INSTANCES);
    this._computeStep = 0;
    this._pingPong = 0;
    this._time = 0;

    if (scenario === 'hybrid') {
      const init = generateInitialBoids(this.entityCount);
      this.wasm.loadBoids(init);
    } else if (scenario === 'gpu-compute') {
      const init = generateInitialBoids(this.entityCount);
      const packed = new Float32Array(this.entityCount * 8);
      for (let i = 0; i < this.entityCount; i++) {
        const o = i * 8;
        packed[o]     = init.px[i];
        packed[o + 1] = init.py[i];
        packed[o + 2] = init.pz[i];
        packed[o + 3] = init.phase[i];
        packed[o + 4] = init.vx[i];
        packed[o + 5] = init.vy[i];
        packed[o + 6] = init.vz[i];
        packed[o + 7] = init.colorSeed[i];
      }
      this.device.queue.writeBuffer(this.computeBufA, 0, packed);
      this.device.queue.writeBuffer(this.computeBufB, 0, packed);
      this._buildComputeBundles(this.entityCount);
    } else if (scenario === 'scene-graph') {
      const init = generateInitialCrystals(this.entityCount);
      this.wasm.loadCrystals(init);
      this._buildCrystalBundle(this.entityCount);
    }
  }

  resize(cssWidth, cssHeight) {
    if (!this.device) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(cssWidth * dpr));
    const h = Math.max(1, Math.round(cssHeight * dpr));
    if (w === this._width && h === this._height && this._msaaTex && this._depthTex) return;

    this._width = w;
    this._height = h;
    this.canvas.width = w;
    this.canvas.height = h;

    this._msaaTex?.destroy();
    this._depthTex?.destroy();

    this._msaaTex = this.device.createTexture({
      size: [w, h],
      sampleCount: this.msaaSamples,
      format: this.format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this._depthTex = this.device.createTexture({
      size: [w, h],
      sampleCount: this.msaaSamples,
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  render(camera, dt) {
    if (!this.initialized || this.disposed) return { cpuMs: 0, gpuMs: -1, visibleCount: 0 };
    const cpuStart = performance.now();
    const safeDt = Math.min(0.05, Math.max(0.001, dt));
    this._time += safeDt;

    // 1. Write FrameUniforms (zero allocations)
    const f = this._frameData;
    f.set(camera.viewProjWebGPU, 0);
    f.set(camera.invViewProjWebGPU, 16);
    f[32] = camera.position[0];
    f[33] = camera.position[1];
    f[34] = camera.position[2];
    f[35] = this._time;
    f[36] = SUN_DIR[0];
    f[37] = SUN_DIR[1];
    f[38] = SUN_DIR[2];
    f[39] = 1.0;
    this.device.queue.writeBuffer(this.frameUniformBuf, 0, f);

    let visibleCount = this.entityCount;

    // 2. Scenario-specific CPU/Wasm update or GPU Compute dispatch prep
    if (this.scenario === 'hybrid') {
      visibleCount = this.wasm.stepBoids(
        this.entityCount,
        safeDt,
        camera.pointerWorld,
        camera.pointerMode,
        BOID_PARAMS.separationWeight,
        BOID_PARAMS.alignmentWeight,
        BOID_PARAMS.cohesionWeight,
        camera.frustumWebGPU,
        true,
      );
      if (visibleCount > 0) {
        // Direct zero-copy slice from Wasm linear memory straight to GPU storage buffer!
        this.device.queue.writeBuffer(
          this.dynamicInstanceBuf,
          0,
          this.wasm.gpuOut,
          0,
          visibleCount * 8,
        );
      }
    } else if (this.scenario === 'scene-graph') {
      visibleCount = this.wasm.stepCrystals(
        this.entityCount,
        this._time,
        camera.frustumWebGPU,
        true,
      );
      this.device.queue.writeBuffer(
        this.dynamicInstanceBuf,
        0,
        this.wasm.gpuOut,
        0,
        this.entityCount * 8,
      );
    }

    const encoder = this.device.createCommandEncoder();
    const useTs = this.hasTimestampQuery && !this._tsPending[this._tsSlot];
    const isCompute = this.scenario === 'gpu-compute';

    // 3. Optional GPU Compute pass (Scenario 2)
    if (isCompute) {
      const peerBudget = Math.min(this.entityCount, COMPUTE_PEER_BUDGET);
      const peerOffset = (this._computeStep * peerBudget) % this.entityCount;
      this._computeStep++;

      this._simU32[0] = this.entityCount;
      this._simU32[1] = peerBudget;
      this._simU32[2] = peerOffset;
      this._simI32[3] = camera.pointerMode;
      this._simF32[4] = safeDt;
      this._simF32[5] = BOID_PARAMS.separationWeight;
      this._simF32[6] = BOID_PARAMS.alignmentWeight;
      this._simF32[7] = BOID_PARAMS.cohesionWeight;
      this._simF32[8] = camera.pointerWorld[0];
      this._simF32[9] = camera.pointerWorld[1];
      this._simF32[10] = camera.pointerWorld[2];
      this._simF32[11] = 0;
      this.device.queue.writeBuffer(this.simUniformBuf, 0, this._simDataBuf);

      const compPassDesc = useTs
        ? { timestampWrites: { querySet: this._tsQuerySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } }
        : undefined;
      const compPass = encoder.beginComputePass(compPassDesc);
      compPass.setPipeline(this.computePipeline);
      compPass.setBindGroup(0, this.computeBGs[this._pingPong]);
      compPass.dispatchWorkgroups(Math.ceil(this.entityCount / 64));
      compPass.end();

      // Flip ping-pong: output is now in buffer (1 - pingPong)
      this._pingPong ^= 1;
    }

    // 4. Main MSAA Render Pass
    const swapView = this.context.getCurrentTexture().createView();
    const renderPassDesc = {
      colorAttachments: [{
        view: this._msaaTex.createView(),
        resolveTarget: swapView,
        clearValue: { r: 0.72, g: 0.80, b: 0.88, a: 1 },
        loadOp: 'clear',
        storeOp: 'discard',
      }],
      depthStencilAttachment: {
        view: this._depthTex.createView(),
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'discard',
      },
    };
    if (useTs) {
      renderPassDesc.timestampWrites = {
        querySet: this._tsQuerySet,
        beginningOfPassWriteIndex: 2,
        endOfPassWriteIndex: 3,
      };
    }

    const pass = encoder.beginRenderPass(renderPassDesc);

    if (this.scenario === 'hybrid') {
      pass.executeBundles([this._arenaBundle]);
      if (visibleCount > 0) {
        pass.setPipeline(this.birdPipeline);
        pass.setBindGroup(0, this.dynamicInstBG);
        pass.setVertexBuffer(0, this.birdVB);
        pass.draw(this.birdVertCount, visibleCount, 0, 0);
      }
      pass.executeBundles([this._backdropBundle]);
    } else if (isCompute) {
      // Zero per-frame draw-call recording: execute pre-recorded Arena -> Compute Birds -> Floor+Sky!
      pass.executeBundles([this._arenaBundle, this._computeBundles[this._pingPong], this._backdropBundle]);
    } else if (this.scenario === 'scene-graph') {
      // Replay N pre-recorded individual draw calls via GPURenderBundle!
      pass.executeBundles([this._arenaBundle, this._crystalBundle, this._backdropBundle]);
    }

    pass.end();

    // 5. Resolve GPU timestamps asynchronously without blocking the frame
    if (useTs) {
      const slot = this._tsSlot;
      const readBuf = this._tsReadBufs[slot];
      const firstQuery = isCompute ? 0 : 2;
      const queryCount = isCompute ? 4 : 2;
      const copyBytes = queryCount * 8;
      encoder.resolveQuerySet(this._tsQuerySet, firstQuery, queryCount, this._tsResolveBuf, 0);
      encoder.copyBufferToBuffer(this._tsResolveBuf, 0, readBuf, 0, copyBytes);
      this.device.queue.submit([encoder.finish()]);

      this._tsPending[slot] = true;
      const hadCompute = isCompute;
      readBuf.mapAsync(GPUMapMode.READ, 0, copyBytes).then(() => {
        if (!this.disposed) {
          const ts = new BigUint64Array(readBuf.getMappedRange(0, copyBytes));
          let ns = 0n;
          if (hadCompute) {
            if (ts[1] >= ts[0]) ns += ts[1] - ts[0];
            if (ts[3] >= ts[2]) ns += ts[3] - ts[2];
          } else {
            if (ts[1] >= ts[0]) ns += ts[1] - ts[0];
          }
          const ms = Number(ns) / 1e6;
          if (ms > 0 && ms < 5000) {
            this.lastGpuMs = ms;
          }
        }
        readBuf.unmap();
        this._tsPending[slot] = false;
      }).catch(() => {
        this._tsPending[slot] = false;
      });
      this._tsSlot ^= 1;
    } else {
      this.device.queue.submit([encoder.finish()]);
    }

    const cpuMs = performance.now() - cpuStart;
    return { cpuMs, gpuMs: this.lastGpuMs, visibleCount };
  }

  dispose() {
    this.disposed = true;
    this.initialized = false;
    this._msaaTex?.destroy();
    this._depthTex?.destroy();
    this.frameUniformBuf?.destroy();
    this.simUniformBuf?.destroy();
    this.dynamicInstanceBuf?.destroy();
    this.computeBufA?.destroy();
    this.computeBufB?.destroy();
    this.arenaVB?.destroy();
    this.birdVB?.destroy();
    this.crystalVB?.destroy();
    this._tsQuerySet?.destroy();
    this._tsResolveBuf?.destroy();
    for (const b of this._tsReadBufs) b?.destroy();
    this.device?.destroy();
  }
}
