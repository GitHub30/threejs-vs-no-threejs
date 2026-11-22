// Official Three.js v0.187.0 TSL (Three Shading Language) shaders & GPU Compute pipeline
// Implements the exact same visual look and peer-budget flocking equations as the Custom WGSL engine.

import { BOID_PARAMS, COMPUTE_PEER_BUDGET, PILLARS, SUN_DIR } from '../shared/config.js';

/**
 * Creates identical daylight atmospheric Sky Dome NodeMaterial in TSL.
 */
export function createTslSkyMaterial(THREE, TSL) {
  const { Fn, positionLocal, normalize, vec3, vec4, float, pow, clamp, mix, dot, max, smoothstep } = TSL;
  const sunDir = vec3(SUN_DIR[0], SUN_DIR[1], SUN_DIR[2]);

  const skyColor = Fn(() => {
    const dir = normalize(positionLocal).toConst();
    const y = dir.y;
    const zenith = vec3(0.045, 0.195, 0.520);
    const horizon = vec3(0.760, 0.825, 0.880);
    const ground = vec3(0.360, 0.440, 0.510);

    const upT = pow(clamp(y.mul(1.65), 0.0, 1.0), 0.58);
    const dnT = pow(clamp(y.negate().mul(1.5), 0.0, 1.0), 0.65);
    const upper = mix(horizon, zenith, upT);
    const lower = mix(horizon, ground, dnT);
    const col = y.greaterThanEqual(0.0).select(upper, lower).toVar();

    const sunDot = max(dot(dir, sunDir), 0.0);
    const halo = pow(sunDot, 10.0).mul(0.24).add(pow(sunDot, 96.0).mul(0.72));
    const disc = smoothstep(0.9985, 0.9994, sunDot).mul(1.9);
    col.addAssign(vec3(1.0, 0.88, 0.68).mul(halo.add(disc)));

    return vec4(col, float(1.0));
  });

  const mat = new THREE.MeshBasicNodeMaterial({
    side: THREE.BackSide,
    depthWrite: false,
    fog: false,
  });
  mat.colorNode = skyColor();
  return mat;
}

/**
 * Creates identical terraced Sanctuary Floor NodeMaterial in TSL (radial stone rings + brass inlay + pillar AO).
 */
export function createTslFloorMaterial(THREE, TSL) {
  const { Fn, positionWorld, cameraPosition, vec2, vec3, vec4, float, length, atan, fract, abs, floor, step, smoothstep, mix, max, min, clamp, dot, pow, normalize, exp } = TSL;
  const sunDir = vec3(SUN_DIR[0], SUN_DIR[1], SUN_DIR[2]);

  const floorColor = Fn(() => {
    const wp = positionWorld;
    const xz = wp.xz;
    const r = length(xz);

    const ang = atan(xz.y, xz.x);
    const ringBand = abs(fract(r.mul(0.085)).sub(0.5));
    const spokeBand = abs(fract(ang.mul(12.0 / 6.2831853)).sub(0.5));
    const ringLine = smoothstep(0.475, 0.498, ringBand);
    const spokeLine = smoothstep(0.482, 0.498, spokeBand).mul(smoothstep(10.0, 18.0, r));
    const tileChecker = step(0.5, fract(floor(r.mul(0.085)).mul(0.5).add(floor(ang.mul(12.0 / 6.2831853)).mul(0.5))));

    const brassRing = smoothstep(0.65, 0.15, abs(r.sub(38.0))).add(smoothstep(0.55, 0.12, abs(r.sub(12.5))));

    const albedo = mix(vec3(0.76, 0.72, 0.65), vec3(0.71, 0.67, 0.61), tileChecker.mul(0.55)).toVar();
    albedo.assign(mix(albedo, vec3(0.54, 0.50, 0.45), max(ringLine, spokeLine).mul(0.35)));
    albedo.assign(mix(albedo, vec3(0.88, 0.62, 0.24), clamp(brassRing, 0.0, 1.0).mul(0.75)));

    const shadowOff = vec2(-SUN_DIR[0] / SUN_DIR[1] * 14.0, -SUN_DIR[2] / SUN_DIR[1] * 14.0);
    const ao = smoothstep(6.5, 16.0, r).toVar();
    const obShad = smoothstep(5.5, 15.0, length(xz.sub(shadowOff.mul(0.75))));
    ao.assign(min(ao, mix(0.62, 1.0, obShad)));

    for (let p = 0; p < PILLARS.length; p++) {
      const pp = vec2(PILLARS[p].x, PILLARS[p].z);
      const dP = length(xz.sub(pp));
      ao.assign(min(ao, mix(0.42, 1.0, smoothstep(3.2, 8.8, dP))));
      const dS = length(xz.sub(pp.add(shadowOff.mul(0.6))));
      ao.assign(min(ao, mix(0.68, 1.0, smoothstep(2.5, 8.5, dS))));
    }

    const N = vec3(0.0, 1.0, 0.0);
    const V = normalize(cameraPosition.sub(wp));
    const ndl = max(dot(N, sunDir), 0.0);
    const H = normalize(sunDir.add(V));
    const spec = pow(max(dot(N, H), 0.0), 42.0).mul(brassRing.mul(0.28).add(0.10));

    const hemi = vec3(0.42, 0.64, 0.92).mul(0.34).add(vec3(0.46, 0.39, 0.32).mul(0.10));
    const lit = albedo.mul(hemi.mul(ao).add(vec3(1.0, 0.94, 0.82).mul(ndl.mul(0.76).mul(ao)))).add(vec3(1.0, 0.93, 0.80).mul(spec.mul(ao))).toVar();

    // Aerial mist + radial blend into sky horizon
    const viewVec = wp.sub(cameraPosition);
    const dist = length(viewVec);
    const viewDir = viewVec.div(max(dist, 0.001));
    const sunBias = pow(max(dot(viewDir, sunDir), 0.0), 6.0);
    const fogCol = mix(vec3(0.76, 0.815, 0.865), vec3(0.95, 0.88, 0.76), sunBias.mul(0.35));
    const fogFactor = float(1.0).sub(exp(pow(dist.mul(0.0022), 2.0).negate()));
    lit.assign(mix(lit, fogCol, clamp(fogFactor, 0.0, 0.88)));

    const edgeFade = smoothstep(105.0, 168.0, r);
    const skyCol = mix(vec3(0.760, 0.825, 0.880), vec3(0.360, 0.440, 0.510), pow(clamp(viewDir.y.negate().mul(1.5), 0.0, 1.0), 0.65));
    lit.assign(mix(lit, skyCol, edgeFade));

    return vec4(lit, float(1.0));
  });

  const mat = new THREE.MeshBasicNodeMaterial({ fog: false });
  mat.colorNode = floorColor();
  return mat;
}

/**
 * Helper TSL function matching birdPalette() from shaders.js identically.
 */
function buildBirdPaletteTSL(TSL) {
  const { Fn, vec3, mix } = TSL;
  return Fn(([seed, region]) => {
    const c0 = vec3(0.82, 0.16, 0.08);
    const c1 = vec3(0.86, 0.46, 0.06);
    const c2 = vec3(0.04, 0.28, 0.38);
    const c3 = vec3(0.90, 0.86, 0.78);
    const base = seed.lessThan(0.28).select(
      c0,
      seed.lessThan(0.54).select(
        c1,
        seed.lessThan(0.80).select(c2, c3),
      ),
    );
    const tipMix = seed.greaterThanEqual(0.80).select(0.35, 0.12);
    const tipCol = mix(base.mul(0.62), vec3(0.95, 0.66, 0.22), tipMix);
    const innerCol = base.mul(1.08);
    const bodyCol = mix(base, vec3(0.96, 0.93, 0.88), 0.18);
    return region.greaterThan(1.5).select(tipCol, region.greaterThan(0.5).select(innerCol, bodyCol));
  });
}

/**
 * Creates TSL Hybrid Bird Material (InstancedMesh + per-instance phase/color attribute).
 */
export function createTslHybridBirdMaterial(THREE, TSL) {
  const { Fn, positionLocal, attribute, sin, abs, max, vec3, float } = TSL;
  const paletteFn = buildBirdPaletteTSL(TSL);

  const aExtra = attribute('aWingExtra', 'vec2');
  const aInstMeta = attribute('aInstMeta', 'vec2'); // x = wingPhase, y = colorSeed

  const wingPos = Fn(() => {
    const p = positionLocal.toVar();
    const w = aExtra.x;
    const flap = sin(aInstMeta.x.sub(abs(w).mul(0.55)));
    p.y.addAssign(flap.mul(w).mul(0.74));
    p.x.mulAssign(float(1.0).sub(abs(flap).mul(max(w, 0.0)).mul(0.10)));
    return p;
  });

  const mat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.42,
    metalness: 0.08,
    side: THREE.DoubleSide,
  });
  mat.positionNode = wingPos();
  mat.colorNode = paletteFn(aInstMeta.y, aExtra.y);
  return mat;
}

/**
 * Creates Scenario 2: Official Three.js TSL GPU Compute Swarm (Ping-Pong Storage Buffers + TSL Vertex Pulling).
 * Evaluates the exact same rotating peerBudget window & physics as BOIDS_COMPUTE_WGSL.
 */
export function createTslComputeSwarm(THREE, TSL, count, init) {
  const {
    Fn, If, Loop, uniform, attribute, instancedArray, instanceIndex,
    positionLocal, normalLocal, transformNormalToView, vec2, vec3, vec4, float, uint, int,
    sin, abs, sqrt, length, dot, cross, normalize, max, clamp, fract,
  } = TSL;

  const posPhaseData = new Float32Array(count * 4);
  const velColorData = new Float32Array(count * 4);
  for (let i = 0; i < count; i++) {
    posPhaseData[i * 4]     = init.px[i];
    posPhaseData[i * 4 + 1] = init.py[i];
    posPhaseData[i * 4 + 2] = init.pz[i];
    posPhaseData[i * 4 + 3] = init.phase[i];
    velColorData[i * 4]     = init.vx[i];
    velColorData[i * 4 + 1] = init.vy[i];
    velColorData[i * 4 + 2] = init.vz[i];
    velColorData[i * 4 + 3] = init.colorSeed[i];
  }

  // Double-buffered storage arrays for race-free ping-pong GPU compute
  const posA = instancedArray(posPhaseData, 'vec4').setName('posPhaseA');
  const velA = instancedArray(velColorData, 'vec4').setName('velColorA');
  const posB = instancedArray(posPhaseData.slice(), 'vec4').setName('posPhaseB');
  const velB = instancedArray(velColorData.slice(), 'vec4').setName('velColorB');

  const peerBudget = Math.min(count, COMPUTE_PEER_BUDGET);

  const uniforms = {
    dt: uniform(0.016),
    peerOffset: uniform(uint(0)),
    ptrMode: uniform(int(0)),
    ptrPos: uniform(new THREE.Vector3()),
    activeBuffer: uniform(int(0)), // 0 = read A, 1 = read B in vertex shader
  };

  const makeComputePass = (srcPos, srcVel, dstPos, dstVel) => Fn(() => {
    const idx = instanceIndex;
    const selfP = srcPos.element(idx);
    const selfV = srcVel.element(idx);

    const pos = selfP.xyz.toVar();
    const phase = selfP.w.toVar();
    const vel = selfV.xyz.toVar();
    const seed = selfV.w;

    const sep = vec3(0.0).toVar();
    const ali = vec3(0.0).toVar();
    const coh = vec3(0.0).toVar();
    const neighbors = float(0.0).toVar();

    const uCount = uint(count);
    const radSq = float(20.25);
    const sepSq = float(4.0);

    Loop({ start: uint(0), end: uint(peerBudget), type: 'uint', condition: '<' }, ({ i }) => {
      const peerIdx = uniforms.peerOffset.add(i).mod(uCount);
      const jPos = srcPos.element(peerIdx).xyz;
      const diff = pos.sub(jPos);
      const d2 = dot(diff, diff);

      If(d2.greaterThan(0.0001).and(d2.lessThan(radSq)), () => {
        neighbors.addAssign(1.0);
        ali.addAssign(srcVel.element(peerIdx).xyz);
        coh.addAssign(jPos);
        If(d2.lessThan(sepSq), () => {
          sep.addAssign(diff.div(d2.add(0.15)));
        });
      });
    });

    const acc = vec3(0.0).toVar();
    If(neighbors.greaterThan(0.0), () => {
      const invN = float(1.0).div(neighbors);
      acc.addAssign(sep.mul(BOID_PARAMS.separationWeight));
      acc.addAssign(ali.mul(invN).sub(vel).mul(BOID_PARAMS.alignmentWeight));
      acc.addAssign(coh.mul(invN).sub(pos).mul(BOID_PARAMS.cohesionWeight));
    });

    // Swirl + central obelisk + outer containment
    const rXzSq = pos.x.mul(pos.x).add(pos.z.mul(pos.z));
    If(rXzSq.greaterThan(0.01), () => {
      const rXz = sqrt(rXzSq);
      const invR = float(1.0).div(rXz);
      acc.x.addAssign(pos.z.negate().mul(invR).mul(2.2));
      acc.z.addAssign(pos.x.mul(invR).mul(2.2));

      If(rXz.lessThan(8.2).and(pos.y.lessThan(24.0)), () => {
        const push = float(8.2).sub(rXz).mul(14.0).mul(invR);
        acc.x.addAssign(pos.x.mul(push));
        acc.z.addAssign(pos.z.mul(push));
      });
      If(rXz.greaterThan(52.0), () => {
        const pull = rXz.sub(52.0).mul(3.2).mul(invR);
        acc.x.subAssign(pos.x.mul(pull));
        acc.z.subAssign(pos.z.mul(pull));
      });
    });

    // 6 Outer Monolith Pillars
    If(pos.y.lessThan(26.0), () => {
      for (let p = 0; p < PILLARS.length; p++) {
        const px = float(PILLARS[p].x);
        const pz = float(PILLARS[p].z);
        const dx = pos.x.sub(px);
        const dz = pos.z.sub(pz);
        const d2 = dx.mul(dx).add(dz.mul(dz));
        If(d2.lessThan(31.36).and(d2.greaterThan(0.0001)), () => {
          const dist = sqrt(d2);
          const push = float(5.6).sub(dist).mul(16.0).div(dist);
          acc.x.addAssign(dx.mul(push));
          acc.z.addAssign(dz.mul(push));
        });
      }
    });

    // Vertical floor/ceiling cushion
    If(pos.y.greaterThan(24.0), () => {
      acc.y.subAssign(pos.y.sub(24.0).mul(4.5));
    }).ElseIf(pos.y.lessThan(-24.0), () => {
      acc.y.addAssign(float(-24.0).sub(pos.y).mul(5.0));
    });
    acc.y.addAssign(float(2.0).sub(pos.y).mul(0.16));

    // Interactive 3D pointer vortex
    If(uniforms.ptrMode.notEqual(int(0)), () => {
      const toPtr = uniforms.ptrPos.sub(pos);
      const d2 = dot(toPtr, toPtr);
      If(d2.lessThan(784.0).and(d2.greaterThan(0.01)), () => {
        const d = sqrt(d2);
        const force = float(uniforms.ptrMode).mul(float(1.0).sub(d.div(28.0))).mul(52.0).div(d);
        acc.addAssign(toPtr.mul(force));
      });
    });

    const dt = clamp(uniforms.dt, 0.0001, 0.05);
    vel.addAssign(acc.mul(dt));
    const spd = length(vel);
    If(spd.greaterThan(22.0), () => {
      vel.mulAssign(float(22.0).div(spd));
    }).ElseIf(spd.lessThan(9.0), () => {
      vel.assign(spd.greaterThan(0.0001).select(vel.mul(float(9.0).div(spd)), vec3(9.0, 0.0, 0.0)));
    });

    pos.addAssign(vel.mul(dt));
    phase.addAssign(dt.mul(float(9.5).add(spd.mul(0.25)).add(max(vel.y, 0.0).mul(0.28))));
    If(phase.greaterThan(62.831853), () => {
      phase.subAssign(62.831853);
    });

    dstPos.element(idx).assign(vec4(pos, phase));
    dstVel.element(idx).assign(vec4(vel, seed));
  })().compute(count);

  const computeAtoB = makeComputePass(posA, velA, posB, velB);
  const computeBtoA = makeComputePass(posB, velB, posA, velA);

  // TSL Lit Material pulling bird position + velocity directly from the active GPU storage buffer
  const paletteFn = buildBirdPaletteTSL(TSL);
  const aExtra = attribute('aWingExtra', 'vec2');

  const computeBirdPos = Fn(() => {
    const useB = uniforms.activeBuffer.equal(int(1));
    const pData = useB.select(posB.element(instanceIndex), posA.element(instanceIndex));
    const vData = useB.select(velB.element(instanceIndex), velA.element(instanceIndex));

    const center = pData.xyz;
    const ph = pData.w;
    const vel = vData.xyz;
    const seed = vData.w;

    const lp = positionLocal.toVar();
    const w = aExtra.x;
    const flap = sin(ph.sub(abs(w).mul(0.55)));
    lp.y.addAssign(flap.mul(w).mul(0.74));
    lp.x.mulAssign(float(1.0).sub(abs(flap).mul(max(w, 0.0)).mul(0.10)));

    const spd = max(length(vel), 0.001);
    const fwd = vel.div(spd);
    const worldUp = abs(fwd.y).greaterThan(0.96).select(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0));
    const right = normalize(cross(worldUp, fwd));
    const up = cross(fwd, right);

    const sc = float(0.88).add(fract(seed.mul(17.31)).mul(0.28));
    return center.add(right.mul(lp.x).add(up.mul(lp.y)).add(fwd.mul(lp.z)).mul(sc));
  });

  const computeBirdColor = Fn(() => {
    const useB = uniforms.activeBuffer.equal(int(1));
    const vData = useB.select(velB.element(instanceIndex), velA.element(instanceIndex));
    return paletteFn(vData.w, aExtra.y);
  });

  const computeBirdNormal = Fn(() => {
    const useB = uniforms.activeBuffer.equal(int(1));
    const vData = useB.select(velB.element(instanceIndex), velA.element(instanceIndex));
    const vel = vData.xyz;
    const spd = max(length(vel), 0.001);
    const fwd = vel.div(spd);
    const worldUp = abs(fwd.y).greaterThan(0.96).select(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0));
    const right = normalize(cross(worldUp, fwd));
    const up = cross(fwd, right);
    const ln = normalLocal;
    const worldNorm = normalize(right.mul(ln.x).add(up.mul(ln.y)).add(fwd.mul(ln.z)));
    return transformNormalToView(worldNorm);
  });

  const material = new THREE.MeshStandardNodeMaterial({
    roughness: 0.42,
    metalness: 0.08,
    side: THREE.DoubleSide,
  });
  material.positionNode = computeBirdPos();
  material.normalNode = computeBirdNormal();
  material.colorNode = computeBirdColor();

  return {
    computeAtoB,
    computeBtoA,
    material,
    uniforms,
    peerBudget,
  };
}
