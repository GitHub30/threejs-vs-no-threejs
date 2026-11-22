// Deterministic Mulberry32 PRNG and identical initial entity states for both engines

export function createPRNG(seed = 20261121) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Generates identical initial 3D bird flock state distributed in graceful aerial rings
 * avoiding the central obelisk and floor.
 */
export function generateInitialBoids(count, seed = 424242) {
  const rand = createPRNG(seed);
  const px = new Float32Array(count);
  const py = new Float32Array(count);
  const pz = new Float32Array(count);
  const vx = new Float32Array(count);
  const vy = new Float32Array(count);
  const vz = new Float32Array(count);
  const phase = new Float32Array(count);
  const colorSeed = new Float32Array(count);

  for (let i = 0; i < count; i++) {
    const angle = rand() * Math.PI * 2;
    const radius = 14.0 + Math.sqrt(rand()) * 42.0;
    const x = Math.cos(angle) * radius;
    const z = Math.sin(angle) * radius;
    const y = (rand() - 0.45) * 34.0;

    // Initial tangential orbital velocity with organic spread
    const speed = 12.0 + rand() * 7.0;
    const tx = -Math.sin(angle);
    const tz = Math.cos(angle);

    px[i] = x;
    py[i] = y;
    pz[i] = z;

    vx[i] = tx * speed + (rand() - 0.5) * 4.5;
    vy[i] = (rand() - 0.5) * 3.5;
    vz[i] = tz * speed + (rand() - 0.5) * 4.5;

    phase[i] = rand() * Math.PI * 2;
    colorSeed[i] = rand();
  }

  return { px, py, pz, vx, vy, vz, phase, colorSeed };
}

/**
 * Generates identical initial orbital crystal beacon parameters for Scenario 3 (Scene Graph Stress).
 */
export function generateInitialCrystals(count, seed = 909090) {
  const rand = createPRNG(seed);
  const radius = new Float32Array(count);
  const angle = new Float32Array(count);
  const baseY = new Float32Array(count);
  const orbitSpd = new Float32Array(count);
  const spinSpd = new Float32Array(count);
  const scale = new Float32Array(count);
  const phase = new Float32Array(count);
  const color = new Float32Array(count);

  for (let i = 0; i < count; i++) {
    const ring = i % 5;
    const baseR = 13.0 + ring * 9.5 + (rand() - 0.5) * 5.2;
    radius[i] = baseR;
    angle[i] = rand() * Math.PI * 2;
    baseY[i] = (rand() - 0.45) * 32.0 + Math.sin(ring * 1.1) * 4.0;
    const dir = ring % 2 === 0 ? 1.0 : -1.0;
    orbitSpd[i] = dir * (0.12 + rand() * 0.24) * (24.0 / (baseR + 8.0));
    spinSpd[i] = (0.7 + rand() * 1.8) * (rand() < 0.5 ? -1 : 1);
    scale[i] = 0.55 + rand() * 0.72;
    phase[i] = rand() * Math.PI * 2;
    color[i] = rand();
  }

  return { radius, angle, baseY, orbitSpd, spinSpd, scale, phase, color };
}
