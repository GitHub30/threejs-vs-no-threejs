import { Matrix3, Matrix4, Sphere, Vector2, Vector3 } from './math';

export type TypedArray = Float32Array | Uint32Array | Uint16Array | Int16Array | Uint8Array | Int8Array;

export interface GeometryGroup {
  start: number;
  count: number;
  materialIndex: number;
}

let nextGeoId = 1;

export class BufferAttribute {
  array: TypedArray;
  itemSize: number;
  normalized: boolean;
  version = 0;

  constructor(array: TypedArray, itemSize: number, normalized = false) {
    this.array = array;
    this.itemSize = itemSize;
    this.normalized = normalized;
  }

  get count(): number {
    return (this.array.length / this.itemSize) | 0;
  }

  set needsUpdate(v: boolean) {
    if (v) this.version++;
  }

  getX(i: number): number { return this.array[i * this.itemSize]; }
  getY(i: number): number { return this.array[i * this.itemSize + 1]; }
  getZ(i: number): number { return this.array[i * this.itemSize + 2]; }

  setX(i: number, x: number): this { this.array[i * this.itemSize] = x; return this; }
  setY(i: number, y: number): this { this.array[i * this.itemSize + 1] = y; return this; }
  setZ(i: number, z: number): this { this.array[i * this.itemSize + 2] = z; return this; }
  setXY(i: number, x: number, y: number): this {
    const o = i * this.itemSize;
    this.array[o] = x; this.array[o + 1] = y;
    return this;
  }
  setXYZ(i: number, x: number, y: number, z: number): this {
    const o = i * this.itemSize;
    this.array[o] = x; this.array[o + 1] = y; this.array[o + 2] = z;
    return this;
  }

  clone(): BufferAttribute {
    const Ctor = this.array.constructor as new (src: TypedArray) => TypedArray;
    return new BufferAttribute(new Ctor(this.array), this.itemSize, this.normalized);
  }
}

export class Float32BufferAttribute extends BufferAttribute {
  constructor(data: ArrayLike<number>, itemSize: number, normalized = false) {
    super(data instanceof Float32Array ? data : new Float32Array(data), itemSize, normalized);
  }
}

export class Uint16BufferAttribute extends BufferAttribute {
  constructor(data: ArrayLike<number>, itemSize: number, normalized = false) {
    super(data instanceof Uint16Array ? data : new Uint16Array(data), itemSize, normalized);
  }
}

const _geoM4 = new Matrix4();
const _geoM3 = new Matrix3();
const _geoV3 = new Vector3();

export class BufferGeometry {
  readonly id = nextGeoId++;
  attributes: Record<string, BufferAttribute> = {};
  index: BufferAttribute | null = null;
  groups: GeometryGroup[] = [];
  boundingSphere: Sphere | null = null;
  _layoutVersion = 1;
  _disposedVersion = 0;
  _onDispose: ((g: BufferGeometry) => void) | null = null;

  setAttribute(name: string, attr: BufferAttribute): this {
    this.attributes[name] = attr;
    this._layoutVersion++;
    return this;
  }

  getAttribute(name: string): BufferAttribute | undefined {
    return this.attributes[name];
  }

  deleteAttribute(name: string): this {
    delete this.attributes[name];
    this._layoutVersion++;
    return this;
  }

  setIndex(idx: BufferAttribute | number[] | Uint16Array | Uint32Array | null): this {
    if (idx === null) {
      this.index = null;
    } else if (idx instanceof BufferAttribute) {
      this.index = idx;
    } else if (idx instanceof Uint16Array || idx instanceof Uint32Array) {
      this.index = new BufferAttribute(idx, 1);
    } else {
      let max = 0;
      for (let i = 0; i < idx.length; i++) if (idx[i] > max) max = idx[i];
      this.index = new BufferAttribute(max > 65535 ? new Uint32Array(idx) : new Uint16Array(idx), 1);
    }
    this._layoutVersion++;
    return this;
  }

  getIndex(): BufferAttribute | null {
    return this.index;
  }

  addGroup(start: number, count: number, materialIndex = 0): void {
    this.groups.push({ start, count, materialIndex });
  }

  clearGroups(): void {
    this.groups.length = 0;
  }

  applyMatrix4(m: Matrix4): this {
    const pos = this.attributes.position;
    if (pos) {
      const a = pos.array;
      for (let i = 0; i < a.length; i += 3) {
        _geoV3.set(a[i], a[i + 1], a[i + 2]).applyMatrix4(m);
        a[i] = _geoV3.x; a[i + 1] = _geoV3.y; a[i + 2] = _geoV3.z;
      }
      pos.version++;
    }
    const nor = this.attributes.normal;
    if (nor && nor.array instanceof Float32Array) {
      _geoM3.getNormalMatrix(m);
      const a = nor.array;
      for (let i = 0; i < a.length; i += 3) {
        _geoV3.set(a[i], a[i + 1], a[i + 2]).applyMatrix3(_geoM3).normalize();
        a[i] = _geoV3.x; a[i + 1] = _geoV3.y; a[i + 2] = _geoV3.z;
      }
      nor.version++;
    }
    if (this.boundingSphere) this.computeBoundingSphere();
    return this;
  }

  applyQuaternion(q: import('./math').Quaternion): this {
    return this.applyMatrix4(_geoM4.makeRotationFromQuaternion(q));
  }

  translate(x: number, y: number, z: number): this {
    return this.applyMatrix4(_geoM4.makeTranslation(x, y, z));
  }
  rotateX(rad: number): this {
    return this.applyMatrix4(_geoM4.makeRotationX(rad));
  }
  rotateY(rad: number): this {
    return this.applyMatrix4(_geoM4.makeRotationY(rad));
  }
  rotateZ(rad: number): this {
    return this.applyMatrix4(_geoM4.makeRotationZ(rad));
  }
  scale(x: number, y: number, z: number): this {
    return this.applyMatrix4(_geoM4.makeScale(x, y, z));
  }

  computeVertexNormals(): this {
    const pos = this.attributes.position;
    if (!pos) return this;
    const p = pos.array;
    const nArr = new Float32Array(p.length);
    const addFace = (ia: number, ib: number, ic: number) => {
      const ax = p[ia], ay = p[ia + 1], az = p[ia + 2];
      const abx = p[ib] - ax, aby = p[ib + 1] - ay, abz = p[ib + 2] - az;
      const acx = p[ic] - ax, acy = p[ic + 1] - ay, acz = p[ic + 2] - az;
      const nx = aby * acz - abz * acy;
      const ny = abz * acx - abx * acz;
      const nz = abx * acy - aby * acx;
      nArr[ia] += nx; nArr[ia + 1] += ny; nArr[ia + 2] += nz;
      nArr[ib] += nx; nArr[ib + 1] += ny; nArr[ib + 2] += nz;
      nArr[ic] += nx; nArr[ic + 1] += ny; nArr[ic + 2] += nz;
    };
    if (this.index) {
      const idx = this.index.array;
      for (let i = 0; i + 2 < idx.length; i += 3) {
        addFace(idx[i] * 3, idx[i + 1] * 3, idx[i + 2] * 3);
      }
    } else {
      for (let i = 0; i + 8 < p.length; i += 9) {
        addFace(i, i + 3, i + 6);
      }
    }
    for (let i = 0; i < nArr.length; i += 3) {
      const l = Math.hypot(nArr[i], nArr[i + 1], nArr[i + 2]);
      if (l > 1e-10) {
        const inv = 1 / l;
        nArr[i] *= inv; nArr[i + 1] *= inv; nArr[i + 2] *= inv;
      } else {
        nArr[i + 1] = 1;
      }
    }
    this.setAttribute('normal', new BufferAttribute(nArr, 3));
    return this;
  }

  computeBoundingSphere(): Sphere {
    const pos = this.attributes.position;
    if (!this.boundingSphere) this.boundingSphere = new Sphere();
    if (!pos || pos.count === 0) {
      this.boundingSphere.center.set(0, 0, 0);
      this.boundingSphere.radius = 0;
      return this.boundingSphere;
    }
    const a = pos.array;
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let i = 0; i < a.length; i += 3) {
      const x = a[i], y = a[i + 1], z = a[i + 2];
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
      if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
    }
    const cx = (minX + maxX) * 0.5, cy = (minY + maxY) * 0.5, cz = (minZ + maxZ) * 0.5;
    let r2 = 0;
    for (let i = 0; i < a.length; i += 3) {
      const dx = a[i] - cx, dy = a[i + 1] - cy, dz = a[i + 2] - cz;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 > r2) r2 = d2;
    }
    this.boundingSphere.center.set(cx, cy, cz);
    this.boundingSphere.radius = Math.sqrt(r2);
    return this.boundingSphere;
  }

  toNonIndexed(): BufferGeometry {
    if (!this.index) return this.clone();
    const out = new BufferGeometry();
    const idx = this.index.array;
    for (const [name, attr] of Object.entries(this.attributes)) {
      const sz = attr.itemSize;
      const Ctor = attr.array.constructor as new (len: number) => TypedArray;
      const dst = new Ctor(idx.length * sz);
      const src = attr.array;
      for (let i = 0; i < idx.length; i++) {
        const s = idx[i] * sz, d = i * sz;
        for (let k = 0; k < sz; k++) dst[d + k] = src[s + k];
      }
      out.setAttribute(name, new BufferAttribute(dst, sz, attr.normalized));
    }
    for (const g of this.groups) out.addGroup(g.start, g.count, g.materialIndex);
    if (this.boundingSphere) out.boundingSphere = this.boundingSphere.clone();
    return out;
  }

  clone(): BufferGeometry {
    const out = new BufferGeometry();
    for (const [name, attr] of Object.entries(this.attributes)) {
      out.setAttribute(name, attr.clone());
    }
    if (this.index) out.setIndex(this.index.clone());
    for (const g of this.groups) out.addGroup(g.start, g.count, g.materialIndex);
    if (this.boundingSphere) out.boundingSphere = this.boundingSphere.clone();
    return out;
  }

  dispose(): void {
    this._disposedVersion++;
    this._onDispose?.(this);
  }
}

// =========================================================================
// Procedural geometry builders
// =========================================================================

export class PlaneGeometry extends BufferGeometry {
  constructor(width = 1, height = 1, widthSegs = 1, heightSegs = 1) {
    super();
    const wx = Math.max(1, Math.floor(widthSegs));
    const wy = Math.max(1, Math.floor(heightSegs));
    const count = (wx + 1) * (wy + 1);
    const pos = new Float32Array(count * 3);
    const nor = new Float32Array(count * 3);
    const uv = new Float32Array(count * 2);
    let vp = 0, vu = 0;
    for (let iy = 0; iy <= wy; iy++) {
      const v = iy / wy;
      const y = (0.5 - v) * height;
      for (let ix = 0; ix <= wx; ix++) {
        const u = ix / wx;
        pos[vp] = (u - 0.5) * width;
        pos[vp + 1] = y;
        pos[vp + 2] = 0;
        nor[vp] = 0; nor[vp + 1] = 0; nor[vp + 2] = 1;
        vp += 3;
        uv[vu++] = u;
        uv[vu++] = 1 - v;
      }
    }
    const idx = new (count > 65535 ? Uint32Array : Uint16Array)(wx * wy * 6);
    let k = 0;
    for (let iy = 0; iy < wy; iy++) {
      for (let ix = 0; ix < wx; ix++) {
        const a = iy * (wx + 1) + ix;
        const b = a + wx + 1;
        const c = b + 1;
        const d = a + 1;
        idx[k++] = a; idx[k++] = b; idx[k++] = d;
        idx[k++] = b; idx[k++] = c; idx[k++] = d;
      }
    }
    this.setAttribute('position', new BufferAttribute(pos, 3));
    this.setAttribute('normal', new BufferAttribute(nor, 3));
    this.setAttribute('uv', new BufferAttribute(uv, 2));
    this.setIndex(new BufferAttribute(idx, 1));
    this.computeBoundingSphere();
  }
}

export class BoxGeometry extends BufferGeometry {
  constructor(width = 1, height = 1, depth = 1, ws = 1, hs = 1, ds = 1) {
    super();
    const pos: number[] = [], nor: number[] = [], uv: number[] = [], idx: number[] = [];
    let base = 0;
    const face = (
      uAxis: number, vAxis: number, wAxis: number,
      uDir: number, vDir: number,
      uLen: number, vLen: number, wPos: number,
      uSegs: number, vSegs: number,
    ) => {
      const n = [0, 0, 0];
      n[wAxis] = wPos > 0 ? 1 : -1;
      for (let iy = 0; iy <= vSegs; iy++) {
        const vy = iy / vSegs;
        for (let ix = 0; ix <= uSegs; ix++) {
          const ux = ix / uSegs;
          const p = [0, 0, 0];
          p[uAxis] = (ux - 0.5) * uLen * uDir;
          p[vAxis] = (vy - 0.5) * vLen * vDir;
          p[wAxis] = wPos;
          pos.push(p[0], p[1], p[2]);
          nor.push(n[0], n[1], n[2]);
          uv.push(ux, 1 - vy);
        }
      }
      for (let iy = 0; iy < vSegs; iy++) {
        for (let ix = 0; ix < uSegs; ix++) {
          const a = base + iy * (uSegs + 1) + ix;
          const b = a + (uSegs + 1);
          const c = b + 1;
          const d = a + 1;
          idx.push(a, b, d, b, c, d);
        }
      }
      base += (uSegs + 1) * (vSegs + 1);
    };
    const hx = width * 0.5, hy = height * 0.5, hz = depth * 0.5;
    face(2, 1, 0, -1, -1, depth, height, hx, ds, hs);  // +X
    face(2, 1, 0, 1, -1, depth, height, -hx, ds, hs);  // -X
    face(0, 2, 1, 1, 1, width, depth, hy, ws, ds);     // +Y
    face(0, 2, 1, 1, -1, width, depth, -hy, ws, ds);   // -Y
    face(0, 1, 2, 1, -1, width, height, hz, ws, hs);   // +Z
    face(0, 1, 2, -1, -1, width, height, -hz, ws, hs); // -Z
    this.setAttribute('position', new Float32BufferAttribute(pos, 3));
    this.setAttribute('normal', new Float32BufferAttribute(nor, 3));
    this.setAttribute('uv', new Float32BufferAttribute(uv, 2));
    this.setIndex(idx);
    this.computeBoundingSphere();
  }
}

/** True multi-segment rounded-bevel box: 1 flat interior quad per face + `segments` smooth quarter-arc corner rings. */
export class RoundedBoxGeometry extends BufferGeometry {
  constructor(width = 1, height = 1, depth = 1, segments = 2, radius = 0.1) {
    super();
    const s = Math.max(1, Math.round(segments));
    const grid = 2 * s + 1;
    const r = Math.min(radius, width * 0.5, height * 0.5, depth * 0.5);
    const ix = Math.max(0, width * 0.5 - r);
    const iy = Math.max(0, height * 0.5 - r);
    const iz = Math.max(0, depth * 0.5 - r);
    const base = new BoxGeometry(1, 1, 1, grid, grid, grid);
    const p = base.attributes.position.array as Float32Array;
    const n = base.attributes.normal.array as Float32Array;
    // Map each discrete grid index k in [0..grid] cleanly:
    //   k = 0..s   -> negative bevel arc (-1 .. 0) around -inner
    //   k = s+1..2s+1 -> positive bevel arc (0 .. +1) around +inner
    const inner = [ix, iy, iz];
    for (let i = 0; i < p.length; i += 3) {
      let nx = 0, ny = 0, nz = 0;
      let bx = 0, by = 0, bz = 0;
      for (let axis = 0; axis < 3; axis++) {
        const k = Math.round((p[i + axis] + 0.5) * grid);
        let anchor: number;
        let dir: number;
        if (k <= s) {
          anchor = -inner[axis];
          dir = -(s - k) / s; // -1 at boundary (k=0) .. 0 at bevel end (k=s)
        } else {
          anchor = inner[axis];
          dir = (k - (s + 1)) / s; // 0 at bevel start (k=s+1) .. +1 at boundary (k=2s+1)
        }
        if (axis === 0) { bx = anchor; nx = dir; }
        else if (axis === 1) { by = anchor; ny = dir; }
        else { bz = anchor; nz = dir; }
      }
      // Spherical normalization of bevel offset vector gives smooth curved corners & edges
      const nl = Math.hypot(nx, ny, nz);
      if (nl > 1e-6) {
        nx /= nl; ny /= nl; nz /= nl;
      } else {
        nx = n[i]; ny = n[i + 1]; nz = n[i + 2];
      }
      p[i] = bx + nx * r;
      p[i + 1] = by + ny * r;
      p[i + 2] = bz + nz * r;
      n[i] = nx; n[i + 1] = ny; n[i + 2] = nz;
    }
    this.setAttribute('position', base.attributes.position);
    this.setAttribute('normal', base.attributes.normal);
    this.setAttribute('uv', base.attributes.uv);
    this.setIndex(base.index);
    this.computeBoundingSphere();
  }
}

export class CylinderGeometry extends BufferGeometry {
  constructor(
    radiusTop = 1,
    radiusBottom = 1,
    height = 1,
    radialSegs = 16,
    heightSegs = 1,
    openEnded = false,
    thetaStart = 0,
    thetaLength = Math.PI * 2,
  ) {
    super();
    const rs = Math.max(3, Math.floor(radialSegs));
    const hs = Math.max(1, Math.floor(heightSegs));
    const pos: number[] = [], nor: number[] = [], uv: number[] = [], idx: number[] = [];
    const halfH = height * 0.5;
    const slope = (radiusBottom - radiusTop) / (height || 1);
    let base = 0;

    // Side torso
    for (let iy = 0; iy <= hs; iy++) {
      const v = iy / hs;
      const r = v * (radiusBottom - radiusTop) + radiusTop;
      const y = halfH - v * height;
      for (let ix = 0; ix <= rs; ix++) {
        const u = ix / rs;
        const theta = thetaStart + u * thetaLength;
        const sinT = Math.sin(theta), cosT = Math.cos(theta);
        pos.push(r * sinT, y, r * cosT);
        const nl = Math.hypot(sinT, slope, cosT) || 1;
        nor.push(sinT / nl, slope / nl, cosT / nl);
        uv.push(u, 1 - v);
      }
    }
    for (let iy = 0; iy < hs; iy++) {
      for (let ix = 0; ix < rs; ix++) {
        const a = base + iy * (rs + 1) + ix;
        const b = a + (rs + 1);
        const c = b + 1;
        const d = a + 1;
        idx.push(a, b, d, b, c, d);
      }
    }
    base += (hs + 1) * (rs + 1);

    // Caps
    const addCap = (top: boolean) => {
      const radius = top ? radiusTop : radiusBottom;
      if (radius <= 0) return;
      const sign = top ? 1 : -1;
      const centerIdx = base++;
      pos.push(0, sign * halfH, 0);
      nor.push(0, sign, 0);
      uv.push(0.5, 0.5);
      const ringStart = base;
      for (let ix = 0; ix <= rs; ix++) {
        const u = ix / rs;
        const theta = thetaStart + u * thetaLength;
        const sinT = Math.sin(theta), cosT = Math.cos(theta);
        pos.push(radius * sinT, sign * halfH, radius * cosT);
        nor.push(0, sign, 0);
        uv.push(cosT * 0.5 + 0.5, sinT * 0.5 * sign + 0.5);
        base++;
      }
      for (let ix = 0; ix < rs; ix++) {
        const a = ringStart + ix, b = a + 1;
        if (top) idx.push(a, b, centerIdx);
        else idx.push(b, a, centerIdx);
      }
    };
    if (!openEnded) {
      addCap(true);
      addCap(false);
    }
    this.setAttribute('position', new Float32BufferAttribute(pos, 3));
    this.setAttribute('normal', new Float32BufferAttribute(nor, 3));
    this.setAttribute('uv', new Float32BufferAttribute(uv, 2));
    this.setIndex(idx);
    this.computeBoundingSphere();
  }
}

export class ConeGeometry extends CylinderGeometry {
  constructor(radius = 1, height = 1, radialSegs = 16, heightSegs = 1, openEnded = false, thetaStart = 0, thetaLength = Math.PI * 2) {
    super(0, radius, height, radialSegs, heightSegs, openEnded, thetaStart, thetaLength);
  }
}

export class SphereGeometry extends BufferGeometry {
  constructor(
    radius = 1,
    widthSegs = 16,
    heightSegs = 12,
    phiStart = 0,
    phiLength = Math.PI * 2,
    thetaStart = 0,
    thetaLength = Math.PI,
  ) {
    super();
    const ws = Math.max(3, Math.floor(widthSegs));
    const hs = Math.max(2, Math.floor(heightSegs));
    const pos: number[] = [], nor: number[] = [], uv: number[] = [], idx: number[] = [];
    for (let iy = 0; iy <= hs; iy++) {
      const v = iy / hs;
      const theta = thetaStart + v * thetaLength;
      const sinT = Math.sin(theta), cosT = Math.cos(theta);
      for (let ix = 0; ix <= ws; ix++) {
        const u = ix / ws;
        const phi = phiStart + u * phiLength;
        const nx = -Math.cos(phi) * sinT;
        const ny = cosT;
        const nz = Math.sin(phi) * sinT;
        pos.push(nx * radius, ny * radius, nz * radius);
        nor.push(nx, ny, nz);
        uv.push(u, 1 - v);
      }
    }
    for (let iy = 0; iy < hs; iy++) {
      for (let ix = 0; ix < ws; ix++) {
        const a = iy * (ws + 1) + ix + 1;
        const b = iy * (ws + 1) + ix;
        const c = (iy + 1) * (ws + 1) + ix;
        const d = (iy + 1) * (ws + 1) + ix + 1;
        if (iy !== 0 || thetaStart > 0) idx.push(a, b, d);
        if (iy !== hs - 1 || thetaStart + thetaLength < Math.PI) idx.push(b, c, d);
      }
    }
    this.setAttribute('position', new Float32BufferAttribute(pos, 3));
    this.setAttribute('normal', new Float32BufferAttribute(nor, 3));
    this.setAttribute('uv', new Float32BufferAttribute(uv, 2));
    this.setIndex(idx);
    this.computeBoundingSphere();
  }
}

export class CircleGeometry extends BufferGeometry {
  constructor(radius = 1, segments = 16, thetaStart = 0, thetaLength = Math.PI * 2) {
    super();
    const seg = Math.max(3, Math.floor(segments));
    const pos: number[] = [0, 0, 0], nor: number[] = [0, 0, 1], uv: number[] = [0.5, 0.5], idx: number[] = [];
    for (let i = 0; i <= seg; i++) {
      const a = thetaStart + (i / seg) * thetaLength;
      const c = Math.cos(a), s = Math.sin(a);
      pos.push(radius * c, radius * s, 0);
      nor.push(0, 0, 1);
      uv.push(c * 0.5 + 0.5, s * 0.5 + 0.5);
    }
    for (let i = 1; i <= seg; i++) idx.push(i, i + 1, 0);
    this.setAttribute('position', new Float32BufferAttribute(pos, 3));
    this.setAttribute('normal', new Float32BufferAttribute(nor, 3));
    this.setAttribute('uv', new Float32BufferAttribute(uv, 2));
    this.setIndex(idx);
    this.computeBoundingSphere();
  }
}

function buildPolyhedron(verts: number[], faces: number[], radius: number, detail: number): BufferGeometry {
  const geo = new BufferGeometry();
  const outPos: number[] = [];
  const pushNorm = (x: number, y: number, z: number) => {
    const l = Math.hypot(x, y, z) || 1;
    outPos.push((x / l) * radius, (y / l) * radius, (z / l) * radius);
  };
  const cols = detail + 1;
  for (let f = 0; f < faces.length; f += 3) {
    const ia = faces[f] * 3, ib = faces[f + 1] * 3, ic = faces[f + 2] * 3;
    const ax = verts[ia], ay = verts[ia + 1], az = verts[ia + 2];
    const bx = verts[ib], by = verts[ib + 1], bz = verts[ib + 2];
    const cx = verts[ic], cy = verts[ic + 1], cz = verts[ic + 2];
    const grid: [number, number, number][][] = [];
    for (let i = 0; i <= cols; i++) {
      const row: [number, number, number][] = [];
      const t = i / cols;
      const lx = ax + (cx - ax) * t, ly = ay + (cy - ay) * t, lz = az + (cz - az) * t;
      const rx = bx + (cx - bx) * t, ry = by + (cy - by) * t, rz = bz + (cz - bz) * t;
      const rowCols = cols - i;
      for (let j = 0; j <= rowCols; j++) {
        if (j === 0 && i === cols) row.push([cx, cy, cz]);
        else {
          const s = rowCols > 0 ? j / rowCols : 0;
          row.push([lx + (rx - lx) * s, ly + (ry - ly) * s, lz + (rz - lz) * s]);
        }
      }
      grid.push(row);
    }
    for (let i = 0; i < cols; i++) {
      for (let j = 0; j < 2 * (cols - i) - 1; j++) {
        const k = (j / 2) | 0;
        if (j % 2 === 0) {
          const p0 = grid[i][k + 1], p1 = grid[i + 1][k], p2 = grid[i][k];
          pushNorm(p0[0], p0[1], p0[2]); pushNorm(p1[0], p1[1], p1[2]); pushNorm(p2[0], p2[1], p2[2]);
        } else {
          const p0 = grid[i][k + 1], p1 = grid[i + 1][k + 1], p2 = grid[i + 1][k];
          pushNorm(p0[0], p0[1], p0[2]); pushNorm(p1[0], p1[1], p1[2]); pushNorm(p2[0], p2[1], p2[2]);
        }
      }
    }
  }
  const posArr = new Float32Array(outPos);
  const norArr = new Float32Array(posArr.length);
  const uvArr = new Float32Array((posArr.length / 3) * 2);
  for (let i = 0, u = 0; i < posArr.length; i += 3, u += 2) {
    const l = Math.hypot(posArr[i], posArr[i + 1], posArr[i + 2]) || 1;
    const nx = posArr[i] / l, ny = posArr[i + 1] / l, nz = posArr[i + 2] / l;
    norArr[i] = nx; norArr[i + 1] = ny; norArr[i + 2] = nz;
    uvArr[u] = Math.atan2(nz, -nx) / (Math.PI * 2) + 0.5;
    uvArr[u + 1] = Math.acos(Math.max(-1, Math.min(1, ny))) / Math.PI;
  }
  geo.setAttribute('position', new BufferAttribute(posArr, 3));
  geo.setAttribute('normal', new BufferAttribute(norArr, 3));
  geo.setAttribute('uv', new BufferAttribute(uvArr, 2));
  geo.computeBoundingSphere();
  return geo;
}

export class IcosahedronGeometry extends BufferGeometry {
  constructor(radius = 1, detail = 0) {
    super();
    const t = (1 + Math.sqrt(5)) * 0.5;
    const verts = [
      -1, t, 0, 1, t, 0, -1, -t, 0, 1, -t, 0,
      0, -1, t, 0, 1, t, 0, -1, -t, 0, 1, -t,
      t, 0, -1, t, 0, 1, -t, 0, -1, -t, 0, 1,
    ];
    const faces = [
      0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11,
      1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
      3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9,
      4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1,
    ];
    const built = buildPolyhedron(verts, faces, radius, detail);
    this.attributes = built.attributes;
    this.boundingSphere = built.boundingSphere;
  }
}

export class OctahedronGeometry extends BufferGeometry {
  constructor(radius = 1, detail = 0) {
    super();
    const verts = [1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1];
    const faces = [0, 2, 4, 0, 4, 3, 0, 3, 5, 0, 5, 2, 1, 2, 5, 1, 5, 3, 1, 3, 4, 1, 4, 2];
    const built = buildPolyhedron(verts, faces, radius, detail);
    this.attributes = built.attributes;
    this.boundingSphere = built.boundingSphere;
  }
}

export class LatheGeometry extends BufferGeometry {
  constructor(points: Vector2[], segments = 16, phiStart = 0, phiLength = Math.PI * 2) {
    super();
    const seg = Math.max(3, Math.floor(segments));
    const n = points.length;
    const pos: number[] = [], uv: number[] = [], idx: number[] = [];
    for (let i = 0; i <= seg; i++) {
      const u = i / seg;
      const phi = phiStart + u * phiLength;
      const sinP = Math.sin(phi), cosP = Math.cos(phi);
      for (let j = 0; j < n; j++) {
        pos.push(points[j].x * sinP, points[j].y, points[j].x * cosP);
        uv.push(u, n > 1 ? j / (n - 1) : 0);
      }
    }
    for (let i = 0; i < seg; i++) {
      for (let j = 0; j < n - 1; j++) {
        const base = j + i * n;
        const a = base, b = base + n, c = base + n + 1, d = base + 1;
        idx.push(a, b, d, b, c, d);
      }
    }
    this.setAttribute('position', new Float32BufferAttribute(pos, 3));
    this.setAttribute('uv', new Float32BufferAttribute(uv, 2));
    this.setIndex(idx);
    this.computeVertexNormals();
    // Weld seam normals when full 360-degree revolution closes onto itself (eliminates hard seam on A321 engines/tires)
    if (Math.abs(phiLength - Math.PI * 2) < 1e-5 && this.attributes.normal) {
      const nor = this.attributes.normal.array as Float32Array;
      const lastOff = seg * n;
      for (let j = 0; j < n; j++) {
        const a = j * 3, b = (lastOff + j) * 3;
        const nx = nor[a] + nor[b], ny = nor[a + 1] + nor[b + 1], nz = nor[a + 2] + nor[b + 2];
        const nl = Math.hypot(nx, ny, nz) || 1;
        nor[a] = nor[b] = nx / nl;
        nor[a + 1] = nor[b + 1] = ny / nl;
        nor[a + 2] = nor[b + 2] = nz / nl;
      }
    }
    this.computeBoundingSphere();
  }
}

// =========================================================================
// Curves & Polygon Triangulation
// =========================================================================

function catmullRom1D(t: number, p0: number, p1: number, p2: number, p3: number): number {
  const v0 = (p2 - p0) * 0.5;
  const v1 = (p3 - p1) * 0.5;
  const t2 = t * t, t3 = t2 * t;
  return (2 * p1 - 2 * p2 + v0 + v1) * t3 + (-3 * p1 + 3 * p2 - 2 * v0 - v1) * t2 + v0 * t + p1;
}

export class SplineCurve {
  points: Vector2[];
  constructor(points: Vector2[] = []) {
    this.points = points;
  }
  getPoint(t: number): Vector2 {
    const pts = this.points;
    const p = (pts.length - 1) * t;
    const idx = Math.floor(p);
    const w = p - idx;
    const p0 = pts[idx === 0 ? 0 : idx - 1];
    const p1 = pts[idx];
    const p2 = pts[Math.min(pts.length - 1, idx + 1)];
    const p3 = pts[Math.min(pts.length - 1, idx + 2)];
    return new Vector2(catmullRom1D(w, p0.x, p1.x, p2.x, p3.x), catmullRom1D(w, p0.y, p1.y, p2.y, p3.y));
  }
  getPoints(divisions = 5): Vector2[] {
    const out: Vector2[] = [];
    for (let i = 0; i <= divisions; i++) out.push(this.getPoint(i / divisions));
    return out;
  }
}

function evalNonUniformCR(t: number, x0: number, x1: number, x2: number, x3: number, dt0: number, dt1: number, dt2: number): number {
  let t1 = (x1 - x0) / dt0 - (x2 - x0) / (dt0 + dt1) + (x2 - x1) / dt1;
  let t2 = (x2 - x1) / dt1 - (x3 - x1) / (dt1 + dt2) + (x3 - x2) / dt2;
  t1 *= dt1;
  t2 *= dt1;
  const c0 = x1;
  const c1 = t1;
  const c2 = -3 * x1 + 3 * x2 - 2 * t1 - t2;
  const c3 = 2 * x1 - 2 * x2 + t1 + t2;
  return ((c3 * t + c2) * t + c1) * t + c0;
}

export class CatmullRomCurve3 {
  points: Vector3[];
  closed: boolean;
  curveType: 'centripetal' | 'chordal' | 'catmullrom';
  private arcCache: number[] | null = null;

  constructor(points: Vector3[], closed = false, curveType: 'centripetal' | 'chordal' | 'catmullrom' = 'centripetal') {
    this.points = points;
    this.closed = closed;
    this.curveType = curveType;
  }

  getPoint(t: number): Vector3 {
    const pts = this.points;
    const l = pts.length;
    const p = (l - (this.closed ? 0 : 1)) * t;
    let idx = Math.floor(p);
    let w = p - idx;
    if (this.closed) {
      idx = ((idx % l) + l) % l;
    } else if (idx >= l - 1) {
      idx = l - 2;
      w = 1;
    }
    const p1 = pts[idx];
    const p2 = pts[(idx + 1) % l];
    let p0: Vector3, p3: Vector3;
    if (this.closed) {
      p0 = pts[(idx - 1 + l) % l];
      p3 = pts[(idx + 2) % l];
    } else {
      p0 = idx > 0 ? pts[idx - 1] : new Vector3(p1.x * 2 - p2.x, p1.y * 2 - p2.y, p1.z * 2 - p2.z);
      p3 = idx + 2 < l ? pts[idx + 2] : new Vector3(p2.x * 2 - p1.x, p2.y * 2 - p1.y, p2.z * 2 - p1.z);
    }
    if (this.curveType === 'catmullrom') {
      return new Vector3(
        catmullRom1D(w, p0.x, p1.x, p2.x, p3.x),
        catmullRom1D(w, p0.y, p1.y, p2.y, p3.y),
        catmullRom1D(w, p0.z, p1.z, p2.z, p3.z),
      );
    }
    const pow = this.curveType === 'chordal' ? 0.5 : 0.25;
    let dt0 = Math.pow(p0.distanceToSquared(p1), pow);
    let dt1 = Math.pow(p1.distanceToSquared(p2), pow);
    let dt2 = Math.pow(p2.distanceToSquared(p3), pow);
    if (dt1 < 1e-4) dt1 = 1;
    if (dt0 < 1e-4) dt0 = dt1;
    if (dt2 < 1e-4) dt2 = dt1;
    return new Vector3(
      evalNonUniformCR(w, p0.x, p1.x, p2.x, p3.x, dt0, dt1, dt2),
      evalNonUniformCR(w, p0.y, p1.y, p2.y, p3.y, dt0, dt1, dt2),
      evalNonUniformCR(w, p0.z, p1.z, p2.z, p3.z, dt0, dt1, dt2),
    );
  }

  private getLengths(divisions = 200): number[] {
    if (this.arcCache && this.arcCache.length === divisions + 1) return this.arcCache;
    const lengths = new Array<number>(divisions + 1);
    let prev = this.getPoint(0);
    let sum = 0;
    lengths[0] = 0;
    for (let i = 1; i <= divisions; i++) {
      const cur = this.getPoint(i / divisions);
      sum += cur.distanceTo(prev);
      lengths[i] = sum;
      prev = cur;
    }
    this.arcCache = lengths;
    return lengths;
  }

  getLength(): number {
    const l = this.getLengths();
    return l[l.length - 1];
  }

  getPointAt(u: number): Vector3 {
    const lens = this.getLengths();
    const total = lens[lens.length - 1];
    if (total <= 0) return this.getPoint(0);
    const target = u * total;
    let lo = 0, hi = lens.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (lens[mid] < target) lo = mid + 1;
      else hi = mid;
    }
    const i = Math.max(1, lo);
    const segLen = lens[i] - lens[i - 1];
    const frac = segLen > 0 ? (target - lens[i - 1]) / segLen : 0;
    return this.getPoint((i - 1 + frac) / (lens.length - 1));
  }

  getSpacedPoints(divisions = 5): Vector3[] {
    const out: Vector3[] = [];
    for (let i = 0; i <= divisions; i++) out.push(this.getPointAt(i / divisions));
    return out;
  }
}

/** Robust ear-clipping 2D polygon triangulation (handles concave Osaka building footprints). */
export const ShapeUtils = {
  triangulateShape(contour: Vector2[], _holes: Vector2[][] = []): number[][] {
    const n = contour.length;
    if (n < 3) return [];
    // Compute signed area (positive = CCW in standard 2D x-right, y-up coordinates)
    let area = 0;
    for (let i = 0, j = n - 1; i < n; j = i++) {
      area += (contour[j].x - contour[i].x) * (contour[j].y + contour[i].y);
    }
    const ccw = area > 0;
    const indices: number[] = [];
    for (let i = 0; i < n; i++) indices.push(ccw ? i : n - 1 - i);

    const cross = (a: Vector2, b: Vector2, c: Vector2) => (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    const ptInTri = (p: Vector2, a: Vector2, b: Vector2, c: Vector2) =>
      cross(a, b, p) >= -1e-9 && cross(b, c, p) >= -1e-9 && cross(c, a, p) >= -1e-9;

    const faces: number[][] = [];
    let guard = indices.length * indices.length + 10;
    while (indices.length > 3 && guard-- > 0) {
      const m = indices.length;
      let clipped = false;
      for (let i = 0; i < m; i++) {
        const iPrev = indices[(i + m - 1) % m];
        const iCur = indices[i];
        const iNext = indices[(i + 1) % m];
        const a = contour[iPrev], b = contour[iCur], c = contour[iNext];
        if (cross(a, b, c) <= 1e-10) continue;
        let hasInside = false;
        for (let k = 0; k < m; k++) {
          const ik = indices[k];
          if (ik === iPrev || ik === iCur || ik === iNext) continue;
          if (ptInTri(contour[ik], a, b, c)) {
            hasInside = true;
            break;
          }
        }
        if (hasInside) continue;
        faces.push([iPrev, iCur, iNext]);
        indices.splice(i, 1);
        clipped = true;
        break;
      }
      if (!clipped) {
        // Fallback fan step for degenerate self-touching footprints
        faces.push([indices[0], indices[1], indices[2]]);
        indices.splice(1, 1);
      }
    }
    if (indices.length === 3) faces.push([indices[0], indices[1], indices[2]]);
    return faces;
  },
};

/** Merge multiple BufferGeometries with matching attribute sets into a single indexed or non-indexed geometry. */
export function mergeGeometries(geometries: BufferGeometry[]): BufferGeometry {
  if (geometries.length === 0) return new BufferGeometry();
  const isIndexed = geometries[0].index !== null;
  const attrNames = Object.keys(geometries[0].attributes);
  const out = new BufferGeometry();

  for (const name of attrNames) {
    const first = geometries[0].attributes[name];
    let totalLen = 0;
    for (const g of geometries) totalLen += g.attributes[name].array.length;
    const Ctor = first.array.constructor as new (len: number) => TypedArray;
    const merged = new Ctor(totalLen);
    let off = 0;
    for (const g of geometries) {
      const src = g.attributes[name].array;
      merged.set(src, off);
      off += src.length;
    }
    out.setAttribute(name, new BufferAttribute(merged, first.itemSize, first.normalized));
  }

  if (isIndexed) {
    let totalIdx = 0;
    let totalVerts = 0;
    for (const g of geometries) {
      totalIdx += g.index ? g.index.count : g.attributes.position.count;
      totalVerts += g.attributes.position.count;
    }
    const idx = new (totalVerts > 65535 ? Uint32Array : Uint16Array)(totalIdx);
    let iOff = 0, vOff = 0;
    for (const g of geometries) {
      if (g.index) {
        const src = g.index.array;
        for (let i = 0; i < src.length; i++) idx[iOff++] = src[i] + vOff;
      } else {
        const vc = g.attributes.position.count;
        for (let i = 0; i < vc; i++) idx[iOff++] = vOff + i;
      }
      vOff += g.attributes.position.count;
    }
    out.setIndex(new BufferAttribute(idx, 1));
  }
  out.computeBoundingSphere();
  return out;
}
