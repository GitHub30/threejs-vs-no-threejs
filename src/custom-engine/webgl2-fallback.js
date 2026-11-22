// Zero-dependency Custom WebGL2 Fallback Renderer paired with the exact same Rust Wasm SIMD128 core.
// Ensures the custom engine still works seamlessly on browsers without WebGPU.

import { BOID_PARAMS, SUN_DIR } from '../shared/config.js';
import {
  createArenaGeometryData,
  createBirdGeometryData,
  createCrystalGeometryData,
} from '../shared/geometry.js';
import { generateInitialBoids, generateInitialCrystals } from '../shared/seed.js';
import { loadWasmCore } from './wasm-bridge.js';

const COMMON_GLSL = `#version 300 es
precision highp float;
vec3 acesToneMap(vec3 c) {
  return clamp((c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14), 0.0, 1.0);
}
vec3 linearToSrgb(vec3 c) {
  return pow(max(c, vec3(0.0)), vec3(1.0 / 2.2));
}
vec3 evalSkyColor(vec3 dir, vec3 sunDir) {
  vec3 zenith = vec3(0.045, 0.195, 0.520);
  vec3 horizon = vec3(0.760, 0.825, 0.880);
  vec3 ground = vec3(0.360, 0.440, 0.510);
  vec3 col = dir.y >= 0.0
    ? mix(horizon, zenith, pow(clamp(dir.y * 1.65, 0.0, 1.0), 0.58))
    : mix(horizon, ground, pow(clamp(-dir.y * 1.5, 0.0, 1.0), 0.65));
  float sd = max(dot(dir, sunDir), 0.0);
  col += vec3(1.0, 0.88, 0.68) * (pow(sd, 10.0) * 0.24 + pow(sd, 96.0) * 0.72 + smoothstep(0.9985, 0.9994, sd) * 1.9);
  return col;
}
vec3 applyAerialFog(vec3 color, float dist, vec3 viewDir, vec3 sunDir) {
  vec3 fogBase = vec3(0.76, 0.815, 0.865);
  float sb = pow(max(dot(viewDir, sunDir), 0.0), 6.0);
  vec3 fogCol = mix(fogBase, vec3(0.95, 0.88, 0.76), sb * 0.35);
  float f = 1.0 - exp(-pow(dist * 0.0022, 2.0));
  return mix(color, fogCol, clamp(f, 0.0, 0.88));
}
vec3 birdPalette(float seed, float region) {
  vec3 base = seed < 0.28 ? vec3(0.82, 0.16, 0.08)
            : seed < 0.54 ? vec3(0.86, 0.46, 0.06)
            : seed < 0.80 ? vec3(0.04, 0.28, 0.38)
            : vec3(0.90, 0.86, 0.78);
  if (region > 1.5) return mix(base * 0.62, vec3(0.95, 0.66, 0.22), seed >= 0.80 ? 0.35 : 0.12);
  if (region > 0.5) return base * 1.08;
  return mix(base, vec3(0.96, 0.93, 0.88), 0.18);
}
`;

const SKY_VS = `#version 300 es
out vec2 vNdc;
void main() {
  vec2 pts[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
  vNdc = pts[gl_VertexID];
  gl_Position = vec4(vNdc, 0.9999, 1.0);
}`;

const SKY_FS = `${COMMON_GLSL}
uniform mat4 uInvVP;
uniform vec3 uCamPos;
uniform vec3 uSunDir;
in vec2 vNdc;
out vec4 fragColor;
void main() {
  vec4 fc = uInvVP * vec4(vNdc, 1.0, 1.0);
  vec3 dir = normalize(fc.xyz / fc.w - uCamPos);
  fragColor = vec4(linearToSrgb(acesToneMap(evalSkyColor(dir, uSunDir))), 1.0);
}`;

const ARENA_VS = `#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNorm;
layout(location=2) in vec3 aCol;
uniform mat4 uVP;
out vec3 vWP;
out vec3 vN;
out vec3 vC;
void main() {
  vWP = aPos; vN = aNorm; vC = aCol;
  gl_Position = uVP * vec4(aPos, 1.0);
}`;

const ARENA_FS = `${COMMON_GLSL}
uniform vec3 uCamPos;
uniform vec3 uSunDir;
in vec3 vWP; in vec3 vN; in vec3 vC;
out vec4 fragColor;
void main() {
  vec3 N = normalize(vN);
  vec3 vv = vWP - uCamPos;
  float dist = length(vv);
  vec3 vd = vv / max(dist, 0.001);
  float ndl = max(dot(N, uSunDir), 0.0);
  vec3 hemi = mix(vec3(0.46, 0.39, 0.32), vec3(0.42, 0.64, 0.92), N.y * 0.5 + 0.5) * 0.55;
  float hao = mix(0.68, 1.0, smoothstep(-44.0, -30.0, vWP.y));
  float isGold = smoothstep(0.25, 0.55, vC.r - vC.b);
  vec3 H = normalize(uSunDir - vd);
  float spec = pow(max(dot(N, H), 0.0), mix(18.0, 56.0, isGold)) * mix(0.08, 0.55, isGold);
  vec3 col = vC * (hemi * hao + vec3(1.0, 0.94, 0.82) * ndl * 1.08) + vec3(1.0, 0.92, 0.72) * spec + vC * isGold * 0.14;
  col = applyAerialFog(col, dist, vd, uSunDir);
  fragColor = vec4(linearToSrgb(acesToneMap(col)), 1.0);
}`;

const BIRD_VS = `#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNorm;
layout(location=2) in vec2 aExtra;
layout(location=3) in vec4 iPosPhase;
layout(location=4) in vec4 iVelColor;
uniform mat4 uVP;
out vec3 vWP; out vec3 vN; out vec3 vAlbedo;
vec3 birdPaletteVS(float seed, float region) {
  vec3 base = seed < 0.28 ? vec3(0.82, 0.16, 0.08)
            : seed < 0.54 ? vec3(0.86, 0.46, 0.06)
            : seed < 0.80 ? vec3(0.04, 0.28, 0.38)
            : vec3(0.90, 0.86, 0.78);
  if (region > 1.5) return mix(base * 0.62, vec3(0.95, 0.66, 0.22), seed >= 0.80 ? 0.35 : 0.12);
  if (region > 0.5) return base * 1.08;
  return mix(base, vec3(0.96, 0.93, 0.88), 0.18);
}
void main() {
  vec3 lp = aPos; vec3 ln = aNorm;
  float w = aExtra.x;
  if (w != 0.0) {
    float flap = sin(iPosPhase.w - abs(w) * 0.55);
    lp.y += flap * w * 0.74;
    lp.x *= 1.0 - abs(flap) * max(w, 0.0) * 0.10;
    ln = normalize(vec3(ln.x + flap * sign(lp.x) * max(w, 0.0) * 0.42, ln.y, ln.z));
  }
  vec3 vel = iVelColor.xyz;
  vec3 fwd = vel / max(length(vel), 0.001);
  vec3 wu = abs(fwd.y) > 0.96 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
  vec3 right = normalize(cross(wu, fwd));
  vec3 up = cross(fwd, right);
  float sc = 0.88 + fract(iVelColor.w * 17.31) * 0.28;
  vWP = iPosPhase.xyz + (right * lp.x + up * lp.y + fwd * lp.z) * sc;
  vN = normalize(right * ln.x + up * ln.y + fwd * ln.z);
  vAlbedo = birdPaletteVS(iVelColor.w, aExtra.y);
  gl_Position = uVP * vec4(vWP, 1.0);
}`;

const BIRD_FS = `${COMMON_GLSL}
uniform vec3 uCamPos;
uniform vec3 uSunDir;
in vec3 vWP; in vec3 vN; in vec3 vAlbedo;
out vec4 fragColor;
void main() {
  vec3 N = normalize(gl_FrontFacing ? vN : -vN);
  vec3 vv = vWP - uCamPos;
  float dist = length(vv);
  vec3 vd = vv / max(dist, 0.001);
  float ndl = max(dot(N, uSunDir), 0.0);
  float wrap = max(0.0, (dot(N, uSunDir) + 0.35) / 1.35) * 0.20;
  vec3 hemi = mix(vec3(0.46, 0.39, 0.32), vec3(0.42, 0.64, 0.92), N.y * 0.5 + 0.5) * 0.56;
  vec3 H = normalize(uSunDir - vd);
  float spec = pow(max(dot(N, H), 0.0), 36.0) * 0.34;
  vec3 col = vAlbedo * (hemi + vec3(1.0, 0.94, 0.82) * (ndl * 1.15 + wrap)) + vec3(1.0, 0.95, 0.86) * spec;
  col = applyAerialFog(col, dist, vd, uSunDir);
  fragColor = vec4(linearToSrgb(acesToneMap(col)), 1.0);
}`;

const CRYSTAL_VS = `#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNorm;
layout(location=2) in vec4 iPosScale;
layout(location=3) in vec4 iRotColor;
uniform mat4 uVP;
out vec3 vWP; out vec3 vN; out vec3 vAlbedo;
vec3 pal(float seed) {
  return seed < 0.28 ? vec3(0.82, 0.16, 0.08)
       : seed < 0.54 ? vec3(0.86, 0.46, 0.06)
       : seed < 0.80 ? vec3(0.04, 0.28, 0.38)
       : vec3(0.90, 0.86, 0.78);
}
void main() {
  float sc = iPosScale.w;
  if (sc <= 0.0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  float cy = cos(iRotColor.x), sy = sin(iRotColor.x);
  float cx = cos(iRotColor.y), sx = sin(iRotColor.y);
  vec3 p1 = vec3(aPos.x, aPos.y * cx - aPos.z * sx, aPos.y * sx + aPos.z * cx);
  vec3 p2 = vec3(p1.x * cy + p1.z * sy, p1.y, -p1.x * sy + p1.z * cy);
  vec3 n1 = vec3(aNorm.x, aNorm.y * cx - aNorm.z * sx, aNorm.y * sx + aNorm.z * cx);
  vec3 n2 = vec3(n1.x * cy + n1.z * sy, n1.y, -n1.x * sy + n1.z * cy);
  vWP = iPosScale.xyz + p2 * sc;
  vN = normalize(n2);
  vAlbedo = pal(iRotColor.z) * 1.08;
  gl_Position = uVP * vec4(vWP, 1.0);
}`;

function compileShader(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(s));
  }
  return s;
}

function linkProgram(gl, vsSrc, fsSrc) {
  const p = gl.createProgram();
  gl.attachShader(p, compileShader(gl, gl.VERTEX_SHADER, vsSrc));
  gl.attachShader(p, compileShader(gl, gl.FRAGMENT_SHADER, fsSrc));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(p));
  }
  return p;
}

export class CustomWebGL2Engine {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = null;
    this.wasm = null;
    this.scenario = 'hybrid';
    this.entityCount = 8192;
    this.initialized = false;
    this.disposed = false;
    this.initMs = 0;
    this.backendLabel = 'WebGL2 (Fallback)';
    this.wasmLabel = '';
    this.hasTimestampQuery = false;
    this._time = 0;
  }

  async init(scenario = 'hybrid', count = 8192) {
    const t0 = performance.now();
    this.wasm = await loadWasmCore();
    this.wasmLabel = this.wasm.label;

    const gl = this.canvas.getContext('webgl2', { antialias: true, powerPreference: 'high-performance' });
    if (!gl) throw new Error('WebGL2 unavailable');
    this.gl = gl;

    this.skyProg = linkProgram(gl, SKY_VS, SKY_FS);
    this.arenaProg = linkProgram(gl, ARENA_VS, ARENA_FS);
    this.birdProg = linkProgram(gl, BIRD_VS, BIRD_FS);
    this.crystalProg = linkProgram(gl, CRYSTAL_VS, BIRD_FS);

    this.skyVao = gl.createVertexArray();

    // Arena VAO
    const arena = createArenaGeometryData();
    this.arenaVertCount = arena.vertexCount;
    const arenaInterleaved = new Float32Array(arena.vertexCount * 9);
    for (let i = 0; i < arena.vertexCount; i++) {
      arenaInterleaved.set(arena.positions.subarray(i * 3, i * 3 + 3), i * 9);
      arenaInterleaved.set(arena.normals.subarray(i * 3, i * 3 + 3), i * 9 + 3);
      arenaInterleaved.set(arena.colors.subarray(i * 3, i * 3 + 3), i * 9 + 6);
    }
    this.arenaVao = gl.createVertexArray();
    gl.bindVertexArray(this.arenaVao);
    this.arenaBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.arenaBuf);
    gl.bufferData(gl.ARRAY_BUFFER, arenaInterleaved, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 36, 0);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 36, 12);
    gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 36, 24);

    // Instance dynamic buffer
    this.instBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instBuf);
    gl.bufferData(gl.ARRAY_BUFFER, 65536 * 32, gl.DYNAMIC_DRAW);

    // Bird VAO
    const bird = createBirdGeometryData();
    this.birdVertCount = bird.vertexCount;
    const birdInterleaved = new Float32Array(bird.vertexCount * 8);
    for (let i = 0; i < bird.vertexCount; i++) {
      birdInterleaved.set(bird.positions.subarray(i * 3, i * 3 + 3), i * 8);
      birdInterleaved.set(bird.normals.subarray(i * 3, i * 3 + 3), i * 8 + 3);
      birdInterleaved.set(bird.wingExtra.subarray(i * 2, i * 2 + 2), i * 8 + 6);
    }
    this.birdVao = gl.createVertexArray();
    gl.bindVertexArray(this.birdVao);
    this.birdBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.birdBuf);
    gl.bufferData(gl.ARRAY_BUFFER, birdInterleaved, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 32, 0);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 32, 12);
    gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 2, gl.FLOAT, false, 32, 24);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instBuf);
    gl.enableVertexAttribArray(3); gl.vertexAttribPointer(3, 4, gl.FLOAT, false, 32, 0); gl.vertexAttribDivisor(3, 1);
    gl.enableVertexAttribArray(4); gl.vertexAttribPointer(4, 4, gl.FLOAT, false, 32, 16); gl.vertexAttribDivisor(4, 1);

    // Crystal VAO
    const crystal = createCrystalGeometryData();
    this.crystalVertCount = crystal.vertexCount;
    const crystalInterleaved = new Float32Array(crystal.vertexCount * 6);
    for (let i = 0; i < crystal.vertexCount; i++) {
      crystalInterleaved.set(crystal.positions.subarray(i * 3, i * 3 + 3), i * 6);
      crystalInterleaved.set(crystal.normals.subarray(i * 3, i * 3 + 3), i * 6 + 3);
    }
    this.crystalVao = gl.createVertexArray();
    gl.bindVertexArray(this.crystalVao);
    this.crystalBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.crystalBuf);
    gl.bufferData(gl.ARRAY_BUFFER, crystalInterleaved, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instBuf);
    gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 4, gl.FLOAT, false, 32, 0); gl.vertexAttribDivisor(2, 1);
    gl.enableVertexAttribArray(3); gl.vertexAttribPointer(3, 4, gl.FLOAT, false, 32, 16); gl.vertexAttribDivisor(3, 1);

    gl.bindVertexArray(null);
    this.resize(this.canvas.clientWidth || 640, this.canvas.clientHeight || 480);
    this.configure(scenario, count);
    this.initMs = performance.now() - t0;
    this.initialized = true;
  }

  configure(scenario, count) {
    this.scenario = scenario;
    this.entityCount = Math.min(count, 65536);
    this._time = 0;
    if (scenario === 'scene-graph') {
      this.wasm.loadCrystals(generateInitialCrystals(this.entityCount));
    } else {
      this.wasm.loadBoids(generateInitialBoids(this.entityCount));
    }
  }

  resize(cssWidth, cssHeight) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = Math.max(1, Math.round(cssWidth * dpr));
    this.canvas.height = Math.max(1, Math.round(cssHeight * dpr));
  }

  render(camera, dt) {
    if (!this.initialized || this.disposed) return { cpuMs: 0, gpuMs: -1, visibleCount: 0 };
    const t0 = performance.now();
    const gl = this.gl;
    const safeDt = Math.min(0.05, Math.max(0.001, dt));
    this._time += safeDt;

    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // Sky
    gl.disable(gl.DEPTH_TEST);
    gl.useProgram(this.skyProg);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.skyProg, 'uInvVP'), false, camera.invViewProjWebGPU);
    gl.uniform3fv(gl.getUniformLocation(this.skyProg, 'uCamPos'), camera.position);
    gl.uniform3fv(gl.getUniformLocation(this.skyProg, 'uSunDir'), SUN_DIR);
    gl.bindVertexArray(this.skyVao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // Arena
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
    gl.useProgram(this.arenaProg);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.arenaProg, 'uVP'), false, camera.viewProjWebGL);
    gl.uniform3fv(gl.getUniformLocation(this.arenaProg, 'uCamPos'), camera.position);
    gl.uniform3fv(gl.getUniformLocation(this.arenaProg, 'uSunDir'), SUN_DIR);
    gl.bindVertexArray(this.arenaVao);
    gl.drawArrays(gl.TRIANGLES, 0, this.arenaVertCount);

    let visibleCount = this.entityCount;
    if (this.scenario === 'scene-graph') {
      visibleCount = this.wasm.stepCrystals(this.entityCount, this._time, camera.frustumWebGL, true);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.instBuf);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.wasm.gpuOut.subarray(0, this.entityCount * 8));
      gl.useProgram(this.crystalProg);
      gl.uniformMatrix4fv(gl.getUniformLocation(this.crystalProg, 'uVP'), false, camera.viewProjWebGL);
      gl.uniform3fv(gl.getUniformLocation(this.crystalProg, 'uCamPos'), camera.position);
      gl.uniform3fv(gl.getUniformLocation(this.crystalProg, 'uSunDir'), SUN_DIR);
      gl.bindVertexArray(this.crystalVao);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, this.crystalVertCount, this.entityCount);
    } else {
      visibleCount = this.wasm.stepBoids(
        this.entityCount, safeDt, camera.pointerWorld, camera.pointerMode,
        BOID_PARAMS.separationWeight, BOID_PARAMS.alignmentWeight, BOID_PARAMS.cohesionWeight,
        camera.frustumWebGL, true,
      );
      if (visibleCount > 0) {
        gl.disable(gl.CULL_FACE);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.instBuf);
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.wasm.gpuOut.subarray(0, visibleCount * 8));
        gl.useProgram(this.birdProg);
        gl.uniformMatrix4fv(gl.getUniformLocation(this.birdProg, 'uVP'), false, camera.viewProjWebGL);
        gl.uniform3fv(gl.getUniformLocation(this.birdProg, 'uCamPos'), camera.position);
        gl.uniform3fv(gl.getUniformLocation(this.birdProg, 'uSunDir'), SUN_DIR);
        gl.bindVertexArray(this.birdVao);
        gl.drawArraysInstanced(gl.TRIANGLES, 0, this.birdVertCount, visibleCount);
      }
    }

    return { cpuMs: performance.now() - t0, gpuMs: -1, visibleCount };
  }

  dispose() {
    this.disposed = true;
    this.initialized = false;
  }
}
