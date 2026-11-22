// Clean, optimized JavaScript implementation of the exact same 3D Spatial-Hash Boids
// physics as wasm/src/lib.rs, used by the Three.js Hybrid scenario for an honest, apples-to-apples comparison.

import { BOID_PARAMS, PILLARS } from './config.js';

const GX = BOID_PARAMS.gx;
const GY = BOID_PARAMS.gy;
const GZ = BOID_PARAMS.gz;
const GRID_CELLS = GX * GY * GZ;
const GRID_MIN_X = BOID_PARAMS.gridMinX;
const GRID_MIN_Y = BOID_PARAMS.gridMinY;
const GRID_MIN_Z = BOID_PARAMS.gridMinZ;
const INV_CELL = 1.0 / BOID_PARAMS.cellSize;
const RAD_SQ = BOID_PARAMS.neighborRadius * BOID_PARAMS.neighborRadius;
const SEP_SQ = BOID_PARAMS.separationRadius * BOID_PARAMS.separationRadius;
const MAX_CELL_CHECKS = BOID_PARAMS.maxCellChecks;
const MIN_SPEED = BOID_PARAMS.minSpeed;
const MAX_SPEED = BOID_PARAMS.maxSpeed;
const OBELISK_R = BOID_PARAMS.obeliskAvoidR;
const PILLAR_R = BOID_PARAMS.pillarAvoidR;
const PILLAR_R_SQ = PILLAR_R * PILLAR_R;

const PILLAR_X = new Float32Array(PILLARS.map((p) => p.x));
const PILLAR_Z = new Float32Array(PILLARS.map((p) => p.z));

export class SpatialBoidsJS {
  constructor(maxCount = 65536) {
    this.maxCount = maxCount;
    this.px = new Float32Array(maxCount);
    this.py = new Float32Array(maxCount);
    this.pz = new Float32Array(maxCount);
    this.vx = new Float32Array(maxCount);
    this.vy = new Float32Array(maxCount);
    this.vz = new Float32Array(maxCount);
    this.phase = new Float32Array(maxCount);
    this.colorSeed = new Float32Array(maxCount);

    this.spx = new Float32Array(maxCount);
    this.spy = new Float32Array(maxCount);
    this.spz = new Float32Array(maxCount);
    this.svx = new Float32Array(maxCount);
    this.svy = new Float32Array(maxCount);
    this.svz = new Float32Array(maxCount);
    this.sphase = new Float32Array(maxCount);
    this.scolor = new Float32Array(maxCount);

    this.cellId = new Uint32Array(maxCount);
    this.cellCount = new Uint32Array(GRID_CELLS);
    this.cellStart = new Uint32Array(GRID_CELLS + 1);
    this.cellCursor = new Uint32Array(GRID_CELLS);
  }

  loadState(init) {
    this.px.set(init.px);
    this.py.set(init.py);
    this.pz.set(init.pz);
    this.vx.set(init.vx);
    this.vy.set(init.vy);
    this.vz.set(init.vz);
    this.phase.set(init.phase);
    this.colorSeed.set(init.colorSeed);
  }

  step(count, dt, ptrX, ptrY, ptrZ, ptrMode, sepW, aliW, cohW) {
    const n = count;
    const safeDt = dt > 0.05 ? 0.05 : dt < 0.0001 ? 0.0001 : dt;

    const {
      px, py, pz, vx, vy, vz, phase, colorSeed,
      spx, spy, spz, svx, svy, svz, sphase, scolor,
      cellId, cellCount, cellStart, cellCursor,
    } = this;

    cellCount.fill(0);

    // 1. Hash birds into 3D uniform grid
    for (let i = 0; i < n; i++) {
      let cx = ((px[i] - GRID_MIN_X) * INV_CELL) | 0;
      let cy = ((py[i] - GRID_MIN_Y) * INV_CELL) | 0;
      let cz = ((pz[i] - GRID_MIN_Z) * INV_CELL) | 0;
      if (cx < 0) cx = 0; else if (cx >= GX) cx = GX - 1;
      if (cy < 0) cy = 0; else if (cy >= GY) cy = GY - 1;
      if (cz < 0) cz = 0; else if (cz >= GZ) cz = GZ - 1;
      const cid = (cy * GZ + cz) * GX + cx;
      cellId[i] = cid;
      cellCount[cid]++;
    }

    // 2. Prefix sum
    let sum = 0;
    for (let c = 0; c < GRID_CELLS; c++) {
      cellStart[c] = sum;
      cellCursor[c] = sum;
      sum += cellCount[c];
    }
    cellStart[GRID_CELLS] = sum;

    // 3. Counting-sort scatter
    for (let i = 0; i < n; i++) {
      const cid = cellId[i];
      const dst = cellCursor[cid]++;
      spx[dst] = px[i];
      spy[dst] = py[i];
      spz[dst] = pz[i];
      svx[dst] = vx[i];
      svy[dst] = vy[i];
      svz[dst] = vz[i];
      sphase[dst] = phase[i];
      scolor[dst] = colorSeed[i];
    }

    // 4. Evaluate 3D flocking + pillar avoidance + bounds + integrate
    for (let i = 0; i < n; i++) {
      const x = spx[i], y = spy[i], z = spz[i];
      let bvx = svx[i], bvy = svy[i], bvz = svz[i];

      let cx = ((x - GRID_MIN_X) * INV_CELL) | 0;
      let cy = ((y - GRID_MIN_Y) * INV_CELL) | 0;
      let cz = ((z - GRID_MIN_Z) * INV_CELL) | 0;
      if (cx < 0) cx = 0; else if (cx >= GX) cx = GX - 1;
      if (cy < 0) cy = 0; else if (cy >= GY) cy = GY - 1;
      if (cz < 0) cz = 0; else if (cz >= GZ) cz = GZ - 1;

      const minX = cx > 0 ? cx - 1 : 0;
      const maxX = cx + 1 < GX ? cx + 1 : GX - 1;
      const minY = cy > 0 ? cy - 1 : 0;
      const maxY = cy + 1 < GY ? cy + 1 : GY - 1;
      const minZ = cz > 0 ? cz - 1 : 0;
      const maxZ = cz + 1 < GZ ? cz + 1 : GZ - 1;

      let sepX = 0, sepY = 0, sepZ = 0;
      let aliX = 0, aliY = 0, aliZ = 0;
      let cohX = 0, cohY = 0, cohZ = 0;
      let neighbors = 0;

      for (let ny = minY; ny <= maxY; ny++) {
        for (let nz = minZ; nz <= maxZ; nz++) {
          const baseRow = (ny * GZ + nz) * GX;
          const jStart = cellStart[baseRow + minX];
          let jEnd = cellStart[baseRow + maxX + 1];
          if (jEnd - jStart > MAX_CELL_CHECKS * 3) {
            jEnd = jStart + MAX_CELL_CHECKS * 3;
          }
          for (let j = jStart; j < jEnd; j++) {
            const jx = spx[j], jy = spy[j], jz = spz[j];
            const dx = x - jx, dy = y - jy, dz = z - jz;
            const d2 = dx * dx + dy * dy + dz * dz;
            if (d2 > 0.0001 && d2 < RAD_SQ) {
              neighbors++;
              aliX += svx[j]; aliY += svy[j]; aliZ += svz[j];
              cohX += jx;     cohY += jy;     cohZ += jz;
              if (d2 < SEP_SQ) {
                const inv = 1.0 / (d2 + 0.15);
                sepX += dx * inv;
                sepY += dy * inv;
                sepZ += dz * inv;
              }
            }
          }
        }
      }

      let ax = 0, ay = 0, az = 0;
      if (neighbors > 0) {
        const invN = 1.0 / neighbors;
        ax += sepX * sepW + (aliX * invN - bvx) * aliW + (cohX * invN - x) * cohW;
        ay += sepY * sepW + (aliY * invN - bvy) * aliW + (cohY * invN - y) * cohW;
        az += sepZ * sepW + (aliZ * invN - bvz) * aliW + (cohZ * invN - z) * cohW;
      }

      const rXzSq = x * x + z * z;
      if (rXzSq > 0.01) {
        const rXz = Math.sqrt(rXzSq);
        const invR = 1.0 / rXz;
        ax += -z * invR * 2.2;
        az += x * invR * 2.2;

        if (rXz < OBELISK_R && y < 24.0) {
          const push = (OBELISK_R - rXz) * 14.0;
          ax += x * invR * push;
          az += z * invR * push;
        }
        if (rXz > 52.0) {
          const over = rXz - 52.0;
          ax -= x * invR * over * 3.2;
          az -= z * invR * over * 3.2;
        }
      }

      for (let p = 0; p < 6; p++) {
        const dx = x - PILLAR_X[p];
        const dz = z - PILLAR_Z[p];
        const d2 = dx * dx + dz * dz;
        if (d2 < PILLAR_R_SQ && d2 > 0.0001 && y < 26.0) {
          const d = Math.sqrt(d2);
          const push = ((PILLAR_R - d) * 16.0) / d;
          ax += dx * push;
          az += dz * push;
        }
      }

      if (y > 24.0) ay -= (y - 24.0) * 4.5;
      else if (y < -24.0) ay += (-24.0 - y) * 5.0;
      ay += (2.0 - y) * 0.16;

      if (ptrMode !== 0) {
        const dx = ptrX - x, dy = ptrY - y, dz = ptrZ - z;
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < 784.0 && d2 > 0.01) {
          const d = Math.sqrt(d2);
          const force = (ptrMode * (1.0 - d / 28.0) * 52.0) / d;
          ax += dx * force;
          ay += dy * force;
          az += dz * force;
        }
      }

      bvx += ax * safeDt;
      bvy += ay * safeDt;
      bvz += az * safeDt;

      const spd = Math.sqrt(bvx * bvx + bvy * bvy + bvz * bvz);
      if (spd > MAX_SPEED) {
        const s = MAX_SPEED / spd;
        bvx *= s; bvy *= s; bvz *= s;
      } else if (spd < MIN_SPEED) {
        if (spd > 0.0001) {
          const s = MIN_SPEED / spd;
          bvx *= s; bvy *= s; bvz *= s;
        } else {
          bvx = MIN_SPEED;
        }
      }

      px[i] = x + bvx * safeDt;
      py[i] = y + bvy * safeDt;
      pz[i] = z + bvz * safeDt;
      vx[i] = bvx;
      vy[i] = bvy;
      vz[i] = bvz;

      const climbBoost = bvy > 0 ? bvy * 0.28 : 0;
      let np = sphase[i] + safeDt * (9.5 + spd * 0.25 + climbBoost);
      if (np > 62.831853) np -= 62.831853;
      phase[i] = np;
      colorSeed[i] = scolor[i];
    }
  }
}
