// Shared procedural 3D geometry builders used identically by Custom Engine and Three.js
import { ARENA_FLOOR_Y, COLORS, OBELISK, PILLARS } from './config.js';

function pushTriangle(outPos, outNorm, outExtra, a, b, c, wa, wb, wc, region) {
  const ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
  const vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
  let nx = uy * vz - uz * vy;
  let ny = uz * vx - ux * vz;
  let nz = ux * vy - uy * vx;
  const len = Math.hypot(nx, ny, nz) || 1.0;
  nx /= len; ny /= len; nz /= len;

  outPos.push(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
  outNorm.push(nx, ny, nz, nx, ny, nz, nx, ny, nz);
  if (outExtra) {
    outExtra.push(wa, region, wb, region, wc, region);
  }
}

/**
 * Sculpted 3D faceted origami bird glider (+Z forward, +Y up, +X right).
 * 10 crisp flat-shaded triangles (30 vertices) with articulated gull-wings.
 * Extra attribute per vertex: (wingWeight, colorRegion).
 *   wingWeight: 0 = fuselage, >0 = wing deflection amplitude, <0 = tail subtle pitch
 *   colorRegion: 0 = fuselage body, 1 = inner wing, 2 = outer wingtip accent
 */
export function createBirdGeometryData() {
  const pos = [];
  const norm = [];
  const extra = [];

  // Key anatomical points (bird nose along +Z)
  const nose      = [ 0.0,   0.04,  1.25];
  const dorsal    = [ 0.0,   0.26,  0.18];
  const belly     = [ 0.0,  -0.18,  0.15];
  const tailRoot  = [ 0.0,   0.04, -0.78];
  const tailL     = [-0.36,  0.10, -1.26];
  const tailR     = [ 0.36,  0.10, -1.26];

  const shoulderL = [-0.24,  0.08,  0.34];
  const shoulderR = [ 0.24,  0.08,  0.34];
  const rootBackL = [-0.18,  0.04, -0.34];
  const rootBackR = [ 0.18,  0.04, -0.34];

  const elbowL    = [-0.88,  0.14,  0.12];
  const elbowR    = [ 0.88,  0.14,  0.12];
  const tipL      = [-1.62,  0.05, -0.38];
  const tipR      = [ 1.62,  0.05, -0.38];

  // Fuselage upper facets (2 tris)
  pushTriangle(pos, norm, extra, nose, dorsal, shoulderL, 0, 0, 0.08, 0);
  pushTriangle(pos, norm, extra, nose, shoulderR, dorsal, 0, 0.08, 0, 0);
  pushTriangle(pos, norm, extra, dorsal, tailRoot, rootBackL, 0, 0, 0.05, 0);
  pushTriangle(pos, norm, extra, dorsal, rootBackR, tailRoot, 0, 0.05, 0, 0);

  // Fuselage lower belly facets (2 tris)
  pushTriangle(pos, norm, extra, nose, shoulderL, belly, 0, 0.08, 0, 0);
  pushTriangle(pos, norm, extra, nose, belly, shoulderR, 0, 0, 0.08, 0);

  // Left Gull-Wing: Inner + Swept Outer Tip (2 tris)
  pushTriangle(pos, norm, extra, shoulderL, rootBackL, elbowL, 0.08, 0.05, 0.52, 1);
  pushTriangle(pos, norm, extra, elbowL, rootBackL, tipL, 0.52, 0.05, 1.0, 2);

  // Right Gull-Wing: Inner + Swept Outer Tip (2 tris)
  pushTriangle(pos, norm, extra, shoulderR, elbowR, rootBackR, 0.08, 0.52, 0.05, 1);
  pushTriangle(pos, norm, extra, elbowR, tipR, rootBackR, 0.52, 1.0, 0.05, 2);

  // Forked Tail Stabilizer (2 tris)
  pushTriangle(pos, norm, extra, tailRoot, tailL, rootBackL, 0, -0.18, 0.05, 2);
  pushTriangle(pos, norm, extra, tailRoot, rootBackR, tailR, 0, 0.05, -0.18, 2);

  // 12 triangles = 36 vertices
  return {
    positions: new Float32Array(pos),
    normals: new Float32Array(norm),
    wingExtra: new Float32Array(extra),
    vertexCount: pos.length / 3,
  };
}

/**
 * Faceted architectural crystal beacon (elongated 8-sided bipyramid = 16 triangles, 48 vertices)
 * Used identically in Scenario 3 (Scene Graph & Draw-Call Stress).
 */
export function createCrystalGeometryData() {
  const pos = [];
  const norm = [];
  const sides = 8;
  const radius = 0.62;
  const topY = 1.45;
  const botY = -1.25;
  const top = [0, topY, 0];
  const bot = [0, botY, 0];

  const ring = [];
  for (let i = 0; i < sides; i++) {
    const a = (i / sides) * Math.PI * 2;
    ring.push([Math.cos(a) * radius, 0.08, Math.sin(a) * radius]);
  }

  for (let i = 0; i < sides; i++) {
    const p0 = ring[i];
    const p1 = ring[(i + 1) % sides];
    pushTriangle(pos, norm, null, top, p1, p0, 0, 0, 0, 0);
    pushTriangle(pos, norm, null, bot, p0, p1, 0, 0, 0, 0);
  }

  return {
    positions: new Float32Array(pos),
    normals: new Float32Array(norm),
    vertexCount: pos.length / 3,
  };
}

function addTaperedPrism(pos, norm, col, cx, cz, y0, y1, r0, r1, sides, colorRGB, angleOffset = 0) {
  const bot = [];
  const top = [];
  for (let i = 0; i < sides; i++) {
    const a = angleOffset + (i / sides) * Math.PI * 2;
    const ca = Math.cos(a), sa = Math.sin(a);
    bot.push([cx + ca * r0, y0, cz + sa * r0]);
    top.push([cx + ca * r1, y1, cz + sa * r1]);
  }

  const pushTriCol = (a, b, c, shadeMul = 1.0) => {
    const ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
    const vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
    let nx = uy * vz - uz * vy;
    let ny = uz * vx - ux * vz;
    let nz = ux * vy - uy * vx;
    const len = Math.hypot(nx, ny, nz) || 1.0;
    nx /= len; ny /= len; nz /= len;

    pos.push(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
    norm.push(nx, ny, nz, nx, ny, nz, nx, ny, nz);
    const cr = colorRGB[0] * shadeMul;
    const cg = colorRGB[1] * shadeMul;
    const cb = colorRGB[2] * shadeMul;
    col.push(cr, cg, cb, cr, cg, cb, cr, cg, cb);
  };

  for (let i = 0; i < sides; i++) {
    const n = (i + 1) % sides;
    pushTriCol(bot[i], top[i], bot[n], 1.0);
    pushTriCol(top[i], top[n], bot[n], 1.0);
    // Top cap
    pushTriCol([cx, y1, cz], top[n], top[i], 1.04);
  }
}

/**
 * Static Sanctuary Architecture: Central Tapered Obelisk + Golden Crown Band + 6 Monolith Pillars
 * Merged into one static flat-shaded vertex-colored mesh.
 */
export function createArenaGeometryData() {
  const pos = [];
  const norm = [];
  const col = [];

  const yFloor = ARENA_FLOOR_Y;
  const stone = COLORS.stoneLight;
  const stoneMid = COLORS.stoneDark;
  const gold = COLORS.goldAccent;

  // Central Sanctuary Obelisk base plinth
  addTaperedPrism(pos, norm, col, 0, 0, yFloor, yFloor + 5.0, 9.2, 7.6, 8, stoneMid, Math.PI / 8);
  // Central Obelisk lower shaft
  addTaperedPrism(pos, norm, col, 0, 0, yFloor + 5.0, 7.0, OBELISK.baseRadius, 2.75, 8, stone, Math.PI / 8);
  // Recessed Golden Resonance Band
  addTaperedPrism(pos, norm, col, 0, 0, 7.0, 10.2, 2.95, 2.95, 8, gold, Math.PI / 8);
  // Central Obelisk upper spire
  addTaperedPrism(pos, norm, col, 0, 0, 10.2, OBELISK.topY, 2.6, OBELISK.topRadius, 8, stone, Math.PI / 8);
  // Pyramid tip
  addTaperedPrism(pos, norm, col, 0, 0, OBELISK.topY, OBELISK.topY + 5.5, OBELISK.topRadius, 0.04, 8, gold, Math.PI / 8);

  // 6 Outer Architectural Monolith Pillars
  for (let i = 0; i < PILLARS.length; i++) {
    const p = PILLARS[i];
    const ang = (i * Math.PI) / 3;
    // Plinth
    addTaperedPrism(pos, norm, col, p.x, p.z, yFloor, yFloor + 3.2, p.radius * 1.65, p.radius * 1.3, 6, stoneMid, ang);
    // Main column
    addTaperedPrism(pos, norm, col, p.x, p.z, yFloor + 3.2, p.topY - 2.2, p.radius, p.radius * 0.72, 6, stone, ang);
    // Golden collar ring
    addTaperedPrism(pos, norm, col, p.x, p.z, p.topY - 2.2, p.topY - 0.6, p.radius * 0.78, p.radius * 0.78, 6, gold, ang);
    // Capstone
    addTaperedPrism(pos, norm, col, p.x, p.z, p.topY - 0.6, p.topY + 1.8, p.radius * 0.72, p.radius * 0.2, 6, stone, ang);
  }

  return {
    positions: new Float32Array(pos),
    normals: new Float32Array(norm),
    colors: new Float32Array(col),
    vertexCount: pos.length / 3,
  };
}
