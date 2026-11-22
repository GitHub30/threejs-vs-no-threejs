// Idiomatic, fairly-optimized Three.js v0.187.0 Engine supporting BOTH:
//   - Three.js WebGPURenderer (vendor/three/three.webgpu.js + vendor/three/three.tsl.js)
//   - Three.js WebGLRenderer  (vendor/three/three.module.js)
// Shares the exact same geometry, deterministic seeds, lighting, sky dome, arena floor, and OrbitCamera.

import { ARENA_FLOOR_Y, BIRD_PALETTE, BOID_PARAMS, PILLARS, SUN_DIR } from '../shared/config.js';
import {
  createArenaGeometryData,
  createBirdGeometryData,
  createCrystalGeometryData,
} from '../shared/geometry.js';
import { generateInitialBoids, generateInitialCrystals } from '../shared/seed.js';
import { SpatialBoidsJS } from '../shared/boids-js.js';
import {
  createTslComputeSwarm,
  createTslFloorMaterial,
  createTslHybridBirdMaterial,
  createTslSkyMaterial,
} from './three-tsl-compute.js';

let cachedWebGLMod = null;
let cachedWebGPUMods = null;

async function loadThreeWebGL() {
  if (!cachedWebGLMod) {
    cachedWebGLMod = await import('../../vendor/three/three.module.js');
  }
  return cachedWebGLMod;
}

async function loadThreeWebGPU() {
  if (!cachedWebGPUMods) {
    const [THREE, TSL] = await Promise.all([
      import('../../vendor/three/three.webgpu.js'),
      import('../../vendor/three/three.tsl.js'),
    ]);
    cachedWebGPUMods = { THREE, TSL };
  }
  return cachedWebGPUMods;
}

// Matching GLSL Sky & Floor shaders for Three.js WebGL2 backend
const WEBGL_SKY_VS = `
varying vec3 vLocalPos;
void main() {
  vLocalPos = position;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;

const WEBGL_SKY_FS = `
uniform vec3 uSunDir;
varying vec3 vLocalPos;
void main() {
  vec3 dir = normalize(vLocalPos);
  vec3 zenith = vec3(0.045, 0.195, 0.520);
  vec3 horizon = vec3(0.760, 0.825, 0.880);
  vec3 ground = vec3(0.360, 0.440, 0.510);
  vec3 col = dir.y >= 0.0
    ? mix(horizon, zenith, pow(clamp(dir.y * 1.65, 0.0, 1.0), 0.58))
    : mix(horizon, ground, pow(clamp(-dir.y * 1.5, 0.0, 1.0), 0.65));
  float sd = max(dot(dir, uSunDir), 0.0);
  col += vec3(1.0, 0.88, 0.68) * (pow(sd, 10.0) * 0.24 + pow(sd, 96.0) * 0.72 + smoothstep(0.9985, 0.9994, sd) * 1.9);
  gl_FragColor = vec4(col, 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`;

const WEBGL_FLOOR_VS = `
varying vec3 vWP;
void main() {
  vec4 wp = modelMatrix * vec4(position, 1.0);
  vWP = wp.xyz;
  gl_Position = projectionMatrix * viewMatrix * wp;
}`;

const WEBGL_FLOOR_FS = `
uniform vec3 uSunDir;
varying vec3 vWP;
void main() {
  vec2 xz = vWP.xz;
  float r = length(xz);
  if (r > 170.0) discard;
  float ang = atan(xz.y, xz.x);
  float ringBand = abs(fract(r * 0.085) - 0.5);
  float spokeBand = abs(fract(ang * (12.0 / 6.2831853)) - 0.5);
  float ringLine = smoothstep(0.475, 0.498, ringBand);
  float spokeLine = smoothstep(0.482, 0.498, spokeBand) * smoothstep(10.0, 18.0, r);
  float tileChecker = step(0.5, fract(floor(r * 0.085) * 0.5 + floor(ang * (12.0 / 6.2831853)) * 0.5));
  float brassRing = smoothstep(0.65, 0.15, abs(r - 38.0)) + smoothstep(0.55, 0.12, abs(r - 12.5));

  vec3 albedo = mix(vec3(0.76, 0.72, 0.65), vec3(0.71, 0.67, 0.61), tileChecker * 0.55);
  albedo = mix(albedo, vec3(0.54, 0.50, 0.45), max(ringLine, spokeLine) * 0.35);
  albedo = mix(albedo, vec3(0.88, 0.62, 0.24), clamp(brassRing, 0.0, 1.0) * 0.75);

  vec2 shadowOff = -uSunDir.xz / max(uSunDir.y, 0.2) * 14.0;
  float ao = smoothstep(6.5, 16.0, r);
  ao = min(ao, mix(0.62, 1.0, smoothstep(5.5, 15.0, length(xz - shadowOff * 0.75))));

  vec2 pPos[6];
  pPos[0] = vec2( 38.0, 0.0);
  pPos[1] = vec2( 19.0, 32.908965);
  pPos[2] = vec2(-19.0, 32.908965);
  pPos[3] = vec2(-38.0, 0.0);
  pPos[4] = vec2(-19.0,-32.908965);
  pPos[5] = vec2( 19.0,-32.908965);
  for (int p = 0; p < 6; p++) {
    ao = min(ao, mix(0.42, 1.0, smoothstep(3.2, 8.8, length(xz - pPos[p]))));
    ao = min(ao, mix(0.68, 1.0, smoothstep(2.5, 8.5, length(xz - (pPos[p] + shadowOff * 0.6)))));
  }

  vec3 N = vec3(0.0, 1.0, 0.0);
  vec3 vv = vWP - cameraPosition;
  float dist = length(vv);
  vec3 vd = vv / max(dist, 0.001);
  float ndl = max(dot(N, uSunDir), 0.0);
  vec3 H = normalize(uSunDir - vd);
  float spec = pow(max(dot(N, H), 0.0), 42.0) * (0.10 + brassRing * 0.28);
  vec3 hemi = vec3(0.42, 0.64, 0.92) * 0.34 + vec3(0.46, 0.39, 0.32) * 0.10;
  vec3 col = albedo * (hemi * ao + vec3(1.0, 0.94, 0.82) * ndl * 0.76 * ao) + vec3(1.0, 0.93, 0.80) * spec * ao;

  float sunBias = pow(max(dot(vd, uSunDir), 0.0), 6.0);
  vec3 fogCol = mix(vec3(0.76, 0.815, 0.865), vec3(0.95, 0.88, 0.76), sunBias * 0.35);
  float f = 1.0 - exp(-pow(dist * 0.0022, 2.0));
  col = mix(col, fogCol, clamp(f, 0.0, 0.88));

  float edgeFade = smoothstep(105.0, 168.0, r);
  vec3 skyCol = mix(vec3(0.760, 0.825, 0.880), vec3(0.360, 0.440, 0.510), pow(clamp(-vd.y * 1.5, 0.0, 1.0), 0.65));
  col = mix(col, skyCol, edgeFade);

  gl_FragColor = vec4(col, 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`;

export class ThreeJSEngine {
  constructor(canvas, preferredBackend = 'webgpu') {
    this.canvas = canvas;
    this.preferredBackend = preferredBackend; // 'webgpu' | 'webgl2'
    this.activeBackend = preferredBackend;

    this.THREE = null;
    this.TSL = null;
    this.renderer = null;
    this.scene = null;
    this.camera = null;

    this.scenario = 'hybrid';
    this.entityCount = 8192;
    this.initialized = false;
    this.disposed = false;

    this.initMs = 0;
    this.backendLabel = 'Three.js';
    this.wasmLabel = 'JS (V8 / JIT)';
    this.hasTimestampQuery = false;
    this.lastGpuMs = -1;
    this._tsPending = false;

    // Scenario objects
    this._boidsJS = new SpatialBoidsJS(65536);
    this._dynamicGroup = null;
    this._birdInstancedMesh = null;
    this._instMetaArray = null;
    this._instMetaAttr = null;

    // GPU Compute (TSL) state
    this._tslCompute = null;
    this._computeStep = 0;
    this._pingPong = 0;
    this._computeFallbackToHybrid = false;

    // Scenario 3 (Individual Meshes in Scene Graph)
    this._crystalMeshes = [];
    this._crystalParams = null;

    this._time = 0;
  }

  async init(scenario = 'hybrid', count = 8192) {
    const t0 = performance.now();
    this.scenario = scenario;
    this.entityCount = count;

    const canUseWebGPU = this.preferredBackend === 'webgpu' && !!navigator.gpu;
    let useWebGPU = false;

    if (canUseWebGPU) {
      try {
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (adapter) {
          const mods = await loadThreeWebGPU();
          this.THREE = mods.THREE;
          this.TSL = mods.TSL;
          this.renderer = new this.THREE.WebGPURenderer({
            canvas: this.canvas,
            antialias: true,
            powerPreference: 'high-performance',
            trackTimestamp: adapter.features.has('timestamp-query'),
          });
          await this.renderer.init();
          useWebGPU = true;
          this.activeBackend = 'webgpu';
          this.hasTimestampQuery = !!this.renderer.backend?.trackTimestamp;
          const arch = adapter.info?.architecture || adapter.info?.vendor || '';
          this.backendLabel = arch ? `Three.js WebGPU (${arch})` : 'Three.js WebGPU';
        }
      } catch (err) {
        console.warn('Three.js WebGPURenderer init failed, falling back to WebGLRenderer:', err);
        this.renderer?.dispose?.();
        this.renderer = null;
      }
    }

    if (!useWebGPU) {
      this.THREE = await loadThreeWebGL();
      this.TSL = null;
      this.renderer = new this.THREE.WebGLRenderer({
        canvas: this.canvas,
        antialias: true,
        powerPreference: 'high-performance',
      });
      this.activeBackend = 'webgl2';
      this.hasTimestampQuery = false;
      this.backendLabel = 'Three.js WebGL2';
    }

    const THREE = this.THREE;
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(new THREE.Color(0.76, 0.815, 0.865), 0.0022);

    this.camera = new THREE.PerspectiveCamera(50, 1.6, 0.5, 650);

    // Identical lighting rig: Directional Sun + Hemisphere Sky/Ground
    const sun = new THREE.DirectionalLight(new THREE.Color(1.0, 0.94, 0.82), 2.3);
    sun.position.set(SUN_DIR[0] * 120, SUN_DIR[1] * 120, SUN_DIR[2] * 120);
    this.scene.add(sun);

    const hemi = new THREE.HemisphereLight(
      new THREE.Color(0.42, 0.64, 0.92),
      new THREE.Color(0.46, 0.39, 0.32),
      1.15,
    );
    this.scene.add(hemi);

    this._buildStaticEnvironment();
    this._dynamicGroup = new THREE.Group();
    this.scene.add(this._dynamicGroup);

    this.resize(this.canvas.clientWidth || 640, this.canvas.clientHeight || 480);
    this.configure(scenario, count);

    this.initMs = performance.now() - t0;
    this.initialized = true;
  }

  _buildStaticEnvironment() {
    const THREE = this.THREE;
    const isWebGPU = this.activeBackend === 'webgpu';

    // 1. Sky Dome
    const skyGeo = new THREE.SphereGeometry(420, 32, 20);
    const skyMat = isWebGPU
      ? createTslSkyMaterial(THREE, this.TSL)
      : new THREE.ShaderMaterial({
          vertexShader: WEBGL_SKY_VS,
          fragmentShader: WEBGL_SKY_FS,
          uniforms: { uSunDir: { value: new THREE.Vector3(SUN_DIR[0], SUN_DIR[1], SUN_DIR[2]) } },
          side: THREE.BackSide,
          depthWrite: false,
          fog: false,
        });
    const skyMesh = new THREE.Mesh(skyGeo, skyMat);
    skyMesh.frustumCulled = false;
    this._skyMesh = skyMesh;
    this.scene.add(skyMesh);

    // 2. Terraced Architectural Sanctuary Floor
    const floorGeo = new THREE.PlaneGeometry(350, 350, 1, 1);
    floorGeo.rotateX(-Math.PI / 2);
    floorGeo.translate(0, ARENA_FLOOR_Y, 0);
    const floorMat = isWebGPU
      ? createTslFloorMaterial(THREE, this.TSL)
      : new THREE.ShaderMaterial({
          vertexShader: WEBGL_FLOOR_VS,
          fragmentShader: WEBGL_FLOOR_FS,
          uniforms: { uSunDir: { value: new THREE.Vector3(SUN_DIR[0], SUN_DIR[1], SUN_DIR[2]) } },
          fog: false,
        });
    const floorMesh = new THREE.Mesh(floorGeo, floorMat);
    floorMesh.frustumCulled = false;
    this.scene.add(floorMesh);

    // 3. Static Monolith Arena (Obelisk + 6 Outer Pillars)
    const arenaData = createArenaGeometryData();
    const arenaGeo = new THREE.BufferGeometry();
    arenaGeo.setAttribute('position', new THREE.Float32BufferAttribute(arenaData.positions, 3));
    arenaGeo.setAttribute('normal', new THREE.Float32BufferAttribute(arenaData.normals, 3));
    arenaGeo.setAttribute('color', new THREE.Float32BufferAttribute(arenaData.colors, 3));
    const arenaMat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.52,
      metalness: 0.14,
    });
    const arenaMesh = new THREE.Mesh(arenaGeo, arenaMat);
    arenaMesh.frustumCulled = false;
    this.scene.add(arenaMesh);
  }

  _clearDynamicScene() {
    while (this._dynamicGroup.children.length > 0) {
      const child = this._dynamicGroup.children.pop();
      if (child.geometry && child.geometry !== this._sharedCrystalGeo) {
        child.geometry.dispose?.();
      }
      if (child.material && !Array.isArray(child.material) && !this._sharedCrystalMats?.includes(child.material)) {
        child.material.dispose?.();
      }
    }
    this._sharedCrystalGeo?.dispose?.();
    this._sharedCrystalGeo = null;
    if (this._sharedCrystalMats) {
      for (const m of this._sharedCrystalMats) m.dispose?.();
      this._sharedCrystalMats = null;
    }
    this._birdInstancedMesh = null;
    this._tslCompute = null;
    this._crystalMeshes = [];
    this._crystalParams = null;
  }

  configure(scenario, count) {
    this.scenario = scenario;
    this.entityCount = count;
    this._time = 0;
    this._computeStep = 0;
    this._pingPong = 0;
    this._computeFallbackToHybrid = false;

    this._clearDynamicScene();

    if (scenario === 'gpu-compute' && this.activeBackend === 'webgpu') {
      this._setupGpuComputeScenario(count);
    } else if (scenario === 'scene-graph') {
      this._setupSceneGraphScenario(count);
    } else {
      if (scenario === 'gpu-compute' && this.activeBackend !== 'webgpu') {
        this._computeFallbackToHybrid = true;
      }
      this._setupHybridScenario(count);
    }
  }

  _createBirdGeometry() {
    const THREE = this.THREE;
    const data = createBirdGeometryData();
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(data.positions, 3));
    geo.setAttribute('normal', new THREE.Float32BufferAttribute(data.normals, 3));
    geo.setAttribute('aWingExtra', new THREE.Float32BufferAttribute(data.wingExtra, 2));
    return geo;
  }

  _createWebGLHybridBirdMaterial() {
    const THREE = this.THREE;
    const mat = new THREE.MeshStandardMaterial({
      roughness: 0.42,
      metalness: 0.08,
      side: THREE.DoubleSide,
    });
    mat.onBeforeCompile = (shader) => {
      shader.vertexShader = shader.vertexShader
        .replace(
          '#include <common>',
          `#include <common>
          attribute vec2 aWingExtra;
          attribute vec2 aInstMeta;
          varying vec3 vBirdAlbedo;
          vec3 birdPal(float seed, float region) {
            vec3 base = seed < 0.28 ? vec3(0.82, 0.16, 0.08)
                      : seed < 0.54 ? vec3(0.86, 0.46, 0.06)
                      : seed < 0.80 ? vec3(0.04, 0.28, 0.38)
                      : vec3(0.90, 0.86, 0.78);
            if (region > 1.5) return mix(base * 0.62, vec3(0.95, 0.66, 0.22), seed >= 0.80 ? 0.35 : 0.12);
            if (region > 0.5) return base * 1.08;
            return mix(base, vec3(0.96, 0.93, 0.88), 0.18);
          }`,
        )
        .replace(
          '#include <begin_vertex>',
          `#include <begin_vertex>
          float w = aWingExtra.x;
          if (w != 0.0) {
            float flap = sin(aInstMeta.x - abs(w) * 0.55);
            transformed.y += flap * w * 0.74;
            transformed.x *= 1.0 - abs(flap) * max(w, 0.0) * 0.10;
          }
          vBirdAlbedo = birdPal(aInstMeta.y, aWingExtra.y);`,
        );

      shader.fragmentShader = shader.fragmentShader
        .replace(
          '#include <common>',
          `#include <common>
          varying vec3 vBirdAlbedo;`,
        )
        .replace(
          '#include <color_fragment>',
          `#include <color_fragment>
          diffuseColor.rgb *= vBirdAlbedo;`,
        );
    };
    return mat;
  }

  _setupHybridScenario(count) {
    const THREE = this.THREE;
    const init = generateInitialBoids(count);
    this._boidsJS.loadState(init);

    const geo = this._createBirdGeometry();
    this._instMetaArray = new Float32Array(count * 2);
    this._instMetaAttr = new THREE.InstancedBufferAttribute(this._instMetaArray, 2);
    this._instMetaAttr.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute('aInstMeta', this._instMetaAttr);

    const mat = this.activeBackend === 'webgpu'
      ? createTslHybridBirdMaterial(THREE, this.TSL)
      : this._createWebGLHybridBirdMaterial();

    const mesh = new THREE.InstancedMesh(geo, mat, count);
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mesh.frustumCulled = false;
    this._birdInstancedMesh = mesh;
    this._dynamicGroup.add(mesh);
  }

  _setupGpuComputeScenario(count) {
    const THREE = this.THREE;
    const init = generateInitialBoids(count);
    const geo = this._createBirdGeometry();
    this._tslCompute = createTslComputeSwarm(THREE, this.TSL, count, init);

    const mesh = new THREE.InstancedMesh(geo, this._tslCompute.material, count);
    mesh.frustumCulled = false;
    mesh.matrixAutoUpdate = false;
    mesh.updateMatrix();
    this._birdInstancedMesh = mesh;
    this._dynamicGroup.add(mesh);
  }

  _setupSceneGraphScenario(count) {
    const THREE = this.THREE;
    const init = generateInitialCrystals(count);
    this._crystalParams = init;

    const cData = createCrystalGeometryData();
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(cData.positions, 3));
    geo.setAttribute('normal', new THREE.Float32BufferAttribute(cData.normals, 3));
    geo.computeBoundingSphere();
    this._sharedCrystalGeo = geo;

    // 4 shared PBR materials matching our curated palette (idiomatic Three.js multi-object scene)
    const mats = BIRD_PALETTE.map((rgb) => new THREE.MeshStandardMaterial({
      color: new THREE.Color(rgb[0] * 1.08, rgb[1] * 1.08, rgb[2] * 1.08),
      roughness: 0.28,
      metalness: 0.22,
    }));
    this._sharedCrystalMats = mats;

    const meshes = new Array(count);
    for (let i = 0; i < count; i++) {
      const seed = init.color[i];
      const matIdx = seed < 0.28 ? 0 : seed < 0.54 ? 1 : seed < 0.80 ? 2 : 3;
      const m = new THREE.Mesh(geo, mats[matIdx]);
      m.scale.setScalar(init.scale[i]);
      const a = init.angle[i];
      m.position.set(Math.cos(a) * init.radius[i], init.baseY[i], Math.sin(a) * init.radius[i]);
      this._dynamicGroup.add(m);
      meshes[i] = m;
    }
    this._crystalMeshes = meshes;
  }

  resize(cssWidth, cssHeight) {
    if (!this.renderer) return;
    const w = Math.max(1, Math.round(cssWidth));
    const h = Math.max(1, Math.round(cssHeight));
    this.renderer.setSize(w, h, false);
    if (this.camera) {
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
    }
  }

  render(orbitCamera, dt) {
    if (!this.initialized || this.disposed) return { cpuMs: 0, gpuMs: -1, visibleCount: 0 };
    const cpuStart = performance.now();
    const safeDt = Math.min(0.05, Math.max(0.001, dt));
    this._time += safeDt;

    orbitCamera.syncThreeCamera(this.camera);
    if (this._skyMesh) {
      this._skyMesh.position.copy(this.camera.position);
    }

    let visibleCount = this.entityCount;

    if (this.scenario === 'gpu-compute' && this._tslCompute && !this._computeFallbackToHybrid) {
      const c = this._tslCompute;
      c.uniforms.dt.value = safeDt;
      c.uniforms.peerOffset.value = (this._computeStep * c.peerBudget) % this.entityCount;
      c.uniforms.ptrMode.value = orbitCamera.pointerMode;
      c.uniforms.ptrPos.value.set(
        orbitCamera.pointerWorld[0],
        orbitCamera.pointerWorld[1],
        orbitCamera.pointerWorld[2],
      );
      this._computeStep++;

      if (this._pingPong === 0) {
        this.renderer.compute(c.computeAtoB);
        c.uniforms.activeBuffer.value = 1;
      } else {
        this.renderer.compute(c.computeBtoA);
        c.uniforms.activeBuffer.value = 0;
      }
      this._pingPong ^= 1;
    } else if (this.scenario === 'scene-graph') {
      const p = this._crystalParams;
      const meshes = this._crystalMeshes;
      const n = meshes.length;
      const t = this._time;
      const fr = this.activeBackend === 'webgpu' ? orbitCamera.frustumWebGPU : orbitCamera.frustumWebGL;
      visibleCount = 0;

      for (let i = 0; i < n; i++) {
        const ang = p.angle[i] + p.orbitSpd[i] * t;
        const x = Math.cos(ang) * p.radius[i];
        const z = Math.sin(ang) * p.radius[i];
        const sb = Math.sin(t * 1.7 + p.phase[i]);
        const y = p.baseY[i] + sb * 2.4;

        const m = meshes[i];
        m.position.set(x, y, z);
        m.rotation.set(sb * 0.6, p.phase[i] + p.spinSpd[i] * t, 0);

        // Track approximate visible count for HUD parity
        if (
          x * fr[0] + y * fr[1] + z * fr[2] + fr[3] + 2.2 >= 0 &&
          x * fr[4] + y * fr[5] + z * fr[6] + fr[7] + 2.2 >= 0 &&
          x * fr[8] + y * fr[9] + z * fr[10] + fr[11] + 2.2 >= 0 &&
          x * fr[12] + y * fr[13] + z * fr[14] + fr[15] + 2.2 >= 0 &&
          x * fr[16] + y * fr[17] + z * fr[18] + fr[19] + 2.2 >= 0 &&
          x * fr[20] + y * fr[21] + z * fr[22] + fr[23] + 2.2 >= 0
        ) {
          visibleCount++;
        }
      }
    } else {
      // Scenario 1: Hybrid JS Spatial-Hash Boids + optimized InstancedMesh direct matrix packing
      const n = this.entityCount;
      this._boidsJS.step(
        n,
        safeDt,
        orbitCamera.pointerWorld[0],
        orbitCamera.pointerWorld[1],
        orbitCamera.pointerWorld[2],
        orbitCamera.pointerMode,
        BOID_PARAMS.separationWeight,
        BOID_PARAMS.alignmentWeight,
        BOID_PARAMS.cohesionWeight,
      );

      const b = this._boidsJS;
      const matArr = this._birdInstancedMesh.instanceMatrix.array;
      const metaArr = this._instMetaArray;
      const fr = this.activeBackend === 'webgpu' ? orbitCamera.frustumWebGPU : orbitCamera.frustumWebGL;
      const boundR = 2.6;
      let vis = 0;

      // Direct column-major 4x4 orthonormal basis construction (zero temporary objects!)
      for (let i = 0; i < n; i++) {
        const x = b.px[i], y = b.py[i], z = b.pz[i];
        if (
          x * fr[0] + y * fr[1] + z * fr[2] + fr[3] + boundR < 0 ||
          x * fr[4] + y * fr[5] + z * fr[6] + fr[7] + boundR < 0 ||
          x * fr[8] + y * fr[9] + z * fr[10] + fr[11] + boundR < 0 ||
          x * fr[12] + y * fr[13] + z * fr[14] + fr[15] + boundR < 0 ||
          x * fr[16] + y * fr[17] + z * fr[18] + fr[19] + boundR < 0 ||
          x * fr[20] + y * fr[21] + z * fr[22] + fr[23] + boundR < 0
        ) {
          continue;
        }

        const vx = b.vx[i], vy = b.vy[i], vz = b.vz[i];
        const invSpd = 1.0 / (Math.hypot(vx, vy, vz) || 1.0);
        const fx = vx * invSpd, fy = vy * invSpd, fz = vz * invSpd;

        // worldUp cross forward
        let rx, ry, rz;
        if (Math.abs(fy) > 0.96) {
          rx = 0; ry = -fz; rz = fy;
        } else {
          rx = fz; ry = 0; rz = -fx;
        }
        const invR = 1.0 / (Math.hypot(rx, ry, rz) || 1.0);
        rx *= invR; ry *= invR; rz *= invR;

        // up = forward cross right
        const ux = fy * rz - fz * ry;
        const uy = fz * rx - fx * rz;
        const uz = fx * ry - fy * rx;

        const seed = b.colorSeed[i];
        const sc = 0.88 + ((seed * 17.31) % 1) * 0.28;
        const o = vis * 16;

        matArr[o]      = rx * sc; matArr[o + 1]  = ry * sc; matArr[o + 2]  = rz * sc; matArr[o + 3]  = 0;
        matArr[o + 4]  = ux * sc; matArr[o + 5]  = uy * sc; matArr[o + 6]  = uz * sc; matArr[o + 7]  = 0;
        matArr[o + 8]  = fx * sc; matArr[o + 9]  = fy * sc; matArr[o + 10] = fz * sc; matArr[o + 11] = 0;
        matArr[o + 12] = x;       matArr[o + 13] = y;       matArr[o + 14] = z;       matArr[o + 15] = 1;

        metaArr[vis * 2]     = b.phase[i];
        metaArr[vis * 2 + 1] = seed;
        vis++;
      }

      visibleCount = vis;
      this._birdInstancedMesh.count = vis;
      this._birdInstancedMesh.instanceMatrix.needsUpdate = true;
      this._instMetaAttr.needsUpdate = true;
    }

    this.renderer.render(this.scene, this.camera);

    // Non-blocking GPU timestamp query readout on Three.js WebGPURenderer
    if (this.hasTimestampQuery && !this._tsPending) {
      this._tsPending = true;
      const isCompute = this.scenario === 'gpu-compute' && this._tslCompute && !this._computeFallbackToHybrid;
      const tasks = isCompute
        ? [this.renderer.resolveTimestampsAsync('render'), this.renderer.resolveTimestampsAsync('compute')]
        : [this.renderer.resolveTimestampsAsync('render')];
      Promise.all(tasks).then((vals) => {
        if (!this.disposed) {
          let total = 0;
          for (const v of vals) {
            if (typeof v === 'number' && Number.isFinite(v) && v > 0 && v < 500) total += v;
          }
          if (total > 0) this.lastGpuMs = total;
        }
        this._tsPending = false;
      }).catch(() => {
        this._tsPending = false;
      });
    }

    const cpuMs = performance.now() - cpuStart;
    return { cpuMs, gpuMs: this.lastGpuMs, visibleCount };
  }

  dispose() {
    this.disposed = true;
    this.initialized = false;
    this.hasTimestampQuery = false;
    this._clearDynamicScene();
    try {
      this.renderer?.dispose?.();
    } catch {
      // Ignore WebGPU async scope teardown errors when switching backends mid-frame
    }
    this.renderer = null;
  }
}
