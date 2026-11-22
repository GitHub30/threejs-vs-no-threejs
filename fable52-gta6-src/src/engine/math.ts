// Compact, zero-dependency 3D math primitives for the custom Osaka City engine.

export const FrontSide = 0;
export const BackSide = 1;
export const DoubleSide = 2;

export const NormalBlending = 1;
export const AdditiveBlending = 2;

export const RepeatWrapping = 1000;
export const ClampToEdgeWrapping = 1001;

export const SRGBColorSpace = 'srgb';
export const NoColorSpace = '';

export const PCFSoftShadowMap = 2;
export const ACESFilmicToneMapping = 4;
export const HalfFloatType = 1016;

export type EulerOrder = 'XYZ' | 'YXZ' | 'ZXY' | 'ZYX' | 'YZX' | 'XZY';

export function srgbToLinearChannel(c: number): number {
  return c <= 0.04045 ? c * (1 / 12.92) : Math.pow((c + 0.055) / 1.055, 2.4);
}

export function linearToSrgbChannel(c: number): number {
  return c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
}

export class Vector2 {
  x: number;
  y: number;
  constructor(x = 0, y = 0) {
    this.x = x;
    this.y = y;
  }
  set(x: number, y: number): this {
    this.x = x;
    this.y = y;
    return this;
  }
  copy(v: Vector2): this {
    this.x = v.x;
    this.y = v.y;
    return this;
  }
  clone(): Vector2 {
    return new Vector2(this.x, this.y);
  }
  add(v: Vector2): this {
    this.x += v.x;
    this.y += v.y;
    return this;
  }
  sub(v: Vector2): this {
    this.x -= v.x;
    this.y -= v.y;
    return this;
  }
  multiplyScalar(s: number): this {
    this.x *= s;
    this.y *= s;
    return this;
  }
  length(): number {
    return Math.hypot(this.x, this.y);
  }
  distanceTo(v: Vector2): number {
    return Math.hypot(this.x - v.x, this.y - v.y);
  }
}

export class Vector3 {
  x: number;
  y: number;
  z: number;
  constructor(x = 0, y = 0, z = 0) {
    this.x = x;
    this.y = y;
    this.z = z;
  }
  set(x: number, y: number, z: number): this {
    this.x = x;
    this.y = y;
    this.z = z;
    return this;
  }
  setScalar(s: number): this {
    this.x = s;
    this.y = s;
    this.z = s;
    return this;
  }
  setX(x: number): this { this.x = x; return this; }
  setY(y: number): this { this.y = y; return this; }
  setZ(z: number): this { this.z = z; return this; }
  fromBufferAttribute(attr: { array: ArrayLike<number>; itemSize: number }, index: number): this {
    const o = index * attr.itemSize;
    this.x = attr.array[o];
    this.y = attr.array[o + 1];
    this.z = attr.itemSize > 2 ? attr.array[o + 2] : 0;
    return this;
  }
  copy(v: Vector3): this {
    this.x = v.x;
    this.y = v.y;
    this.z = v.z;
    return this;
  }
  clone(): Vector3 {
    return new Vector3(this.x, this.y, this.z);
  }
  fromArray(arr: ArrayLike<number>, offset = 0): this {
    this.x = arr[offset];
    this.y = arr[offset + 1];
    this.z = arr[offset + 2];
    return this;
  }
  toArray(arr: number[] | Float32Array = [], offset = 0): number[] | Float32Array {
    arr[offset] = this.x;
    arr[offset + 1] = this.y;
    arr[offset + 2] = this.z;
    return arr;
  }
  add(v: Vector3): this {
    this.x += v.x;
    this.y += v.y;
    this.z += v.z;
    return this;
  }
  addVectors(a: Vector3, b: Vector3): this {
    this.x = a.x + b.x;
    this.y = a.y + b.y;
    this.z = a.z + b.z;
    return this;
  }
  addScaledVector(v: Vector3, s: number): this {
    this.x += v.x * s;
    this.y += v.y * s;
    this.z += v.z * s;
    return this;
  }
  sub(v: Vector3): this {
    this.x -= v.x;
    this.y -= v.y;
    this.z -= v.z;
    return this;
  }
  subVectors(a: Vector3, b: Vector3): this {
    this.x = a.x - b.x;
    this.y = a.y - b.y;
    this.z = a.z - b.z;
    return this;
  }
  multiplyScalar(s: number): this {
    this.x *= s;
    this.y *= s;
    this.z *= s;
    return this;
  }
  divideScalar(s: number): this {
    return this.multiplyScalar(s !== 0 ? 1 / s : 0);
  }
  dot(v: Vector3): number {
    return this.x * v.x + this.y * v.y + this.z * v.z;
  }
  cross(v: Vector3): this {
    return this.crossVectors(this, v);
  }
  crossVectors(a: Vector3, b: Vector3): this {
    const ax = a.x, ay = a.y, az = a.z;
    const bx = b.x, by = b.y, bz = b.z;
    this.x = ay * bz - az * by;
    this.y = az * bx - ax * bz;
    this.z = ax * by - ay * bx;
    return this;
  }
  lengthSq(): number {
    return this.x * this.x + this.y * this.y + this.z * this.z;
  }
  length(): number {
    return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z);
  }
  normalize(): this {
    const l = this.length();
    return l > 0 ? this.multiplyScalar(1 / l) : this;
  }
  distanceToSquared(v: Vector3): number {
    const dx = this.x - v.x, dy = this.y - v.y, dz = this.z - v.z;
    return dx * dx + dy * dy + dz * dz;
  }
  distanceTo(v: Vector3): number {
    return Math.sqrt(this.distanceToSquared(v));
  }
  lerp(v: Vector3, alpha: number): this {
    this.x += (v.x - this.x) * alpha;
    this.y += (v.y - this.y) * alpha;
    this.z += (v.z - this.z) * alpha;
    return this;
  }
  lerpVectors(a: Vector3, b: Vector3, alpha: number): this {
    this.x = a.x + (b.x - a.x) * alpha;
    this.y = a.y + (b.y - a.y) * alpha;
    this.z = a.z + (b.z - a.z) * alpha;
    return this;
  }
  negate(): this {
    this.x = -this.x;
    this.y = -this.y;
    this.z = -this.z;
    return this;
  }
  applyMatrix3(m: Matrix3): this {
    const e = m.elements;
    const x = this.x, y = this.y, z = this.z;
    this.x = e[0] * x + e[3] * y + e[6] * z;
    this.y = e[1] * x + e[4] * y + e[7] * z;
    this.z = e[2] * x + e[5] * y + e[8] * z;
    return this;
  }
  applyMatrix4(m: Matrix4): this {
    const e = m.elements;
    const x = this.x, y = this.y, z = this.z;
    const w = 1 / (e[3] * x + e[7] * y + e[11] * z + e[15]);
    this.x = (e[0] * x + e[4] * y + e[8] * z + e[12]) * w;
    this.y = (e[1] * x + e[5] * y + e[9] * z + e[13]) * w;
    this.z = (e[2] * x + e[6] * y + e[10] * z + e[14]) * w;
    return this;
  }
  applyQuaternion(q: Quaternion): this {
    const x = this.x, y = this.y, z = this.z;
    const qx = q.x, qy = q.y, qz = q.z, qw = q.w;
    const ix = qw * x + qy * z - qz * y;
    const iy = qw * y + qz * x - qx * z;
    const iz = qw * z + qx * y - qy * x;
    const iw = -qx * x - qy * y - qz * z;
    this.x = ix * qw + iw * -qx + iy * -qz - iz * -qy;
    this.y = iy * qw + iw * -qy + iz * -qx - ix * -qz;
    this.z = iz * qw + iw * -qz + ix * -qy - iy * -qx;
    return this;
  }
}

export class Vector4 {
  x: number;
  y: number;
  z: number;
  w: number;
  constructor(x = 0, y = 0, z = 0, w = 1) {
    this.x = x;
    this.y = y;
    this.z = z;
    this.w = w;
  }
  set(x: number, y: number, z: number, w: number): this {
    this.x = x;
    this.y = y;
    this.z = z;
    this.w = w;
    return this;
  }
  copy(v: Vector4): this {
    this.x = v.x;
    this.y = v.y;
    this.z = v.z;
    this.w = v.w;
    return this;
  }
}

export type ColorLike = Color | number | string;

export class Color {
  // Stored in linear working color space (matching modern PBR workflow)
  r = 1;
  g = 1;
  b = 1;

  constructor(r?: ColorLike, g?: number, b?: number) {
    if (r === undefined) return;
    if (g !== undefined && b !== undefined) {
      this.setRGB(r as number, g, b);
    } else {
      this.set(r);
    }
  }

  set(v: ColorLike): this {
    if (v instanceof Color) return this.copy(v);
    if (typeof v === 'number') return this.setHex(v);
    if (typeof v === 'string') {
      const hex = v.startsWith('#') ? parseInt(v.slice(1), 16) : parseInt(v, 16);
      if (!Number.isNaN(hex)) return this.setHex(hex);
    }
    return this;
  }

  setHex(hex: number): this {
    const h = Math.floor(hex);
    this.r = srgbToLinearChannel(((h >> 16) & 255) / 255);
    this.g = srgbToLinearChannel(((h >> 8) & 255) / 255);
    this.b = srgbToLinearChannel((h & 255) / 255);
    return this;
  }

  getHex(): number {
    const r = Math.max(0, Math.min(255, Math.round(linearToSrgbChannel(this.r) * 255)));
    const g = Math.max(0, Math.min(255, Math.round(linearToSrgbChannel(this.g) * 255)));
    const b = Math.max(0, Math.min(255, Math.round(linearToSrgbChannel(this.b) * 255)));
    return (r << 16) | (g << 8) | b;
  }

  setRGB(r: number, g: number, b: number): this {
    this.r = r;
    this.g = g;
    this.b = b;
    return this;
  }

  setHSL(h: number, s: number, l: number): this {
    const hue = ((h % 1) + 1) % 1;
    let sr: number, sg: number, sb: number;
    if (s === 0) {
      sr = sg = sb = l;
    } else {
      const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
      const p = 2 * l - q;
      const f = (t: number) => {
        if (t < 0) t += 1;
        if (t > 1) t -= 1;
        if (t < 1 / 6) return p + (q - p) * 6 * t;
        if (t < 0.5) return q;
        if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
        return p;
      };
      sr = f(hue + 1 / 3);
      sg = f(hue);
      sb = f(hue - 1 / 3);
    }
    // Match Three.js r152+ workingColorSpace behavior: setHSL writes directly to linear RGB
    this.r = sr;
    this.g = sg;
    this.b = sb;
    return this;
  }

  copy(c: Color): this {
    this.r = c.r;
    this.g = c.g;
    this.b = c.b;
    return this;
  }

  clone(): Color {
    const c = new Color();
    c.r = this.r;
    c.g = this.g;
    c.b = this.b;
    return c;
  }

  multiplyScalar(s: number): this {
    this.r *= s;
    this.g *= s;
    this.b *= s;
    return this;
  }

  lerp(c: Color, alpha: number): this {
    this.r += (c.r - this.r) * alpha;
    this.g += (c.g - this.g) * alpha;
    this.b += (c.b - this.b) * alpha;
    return this;
  }
}

export class Euler {
  private _x: number;
  private _y: number;
  private _z: number;
  private _order: EulerOrder;
  _onChange: (() => void) | null = null;

  constructor(x = 0, y = 0, z = 0, order: EulerOrder = 'XYZ') {
    this._x = x;
    this._y = y;
    this._z = z;
    this._order = order;
  }
  get x() { return this._x; }
  set x(v: number) { if (this._x !== v) { this._x = v; this._onChange?.(); } }
  get y() { return this._y; }
  set y(v: number) { if (this._y !== v) { this._y = v; this._onChange?.(); } }
  get z() { return this._z; }
  set z(v: number) { if (this._z !== v) { this._z = v; this._onChange?.(); } }
  get order() { return this._order; }
  set order(v: EulerOrder) { if (this._order !== v) { this._order = v; this._onChange?.(); } }

  set(x: number, y: number, z: number, order?: EulerOrder): this {
    this._x = x;
    this._y = y;
    this._z = z;
    if (order) this._order = order;
    this._onChange?.();
    return this;
  }
  copy(e: Euler): this {
    this._x = e._x;
    this._y = e._y;
    this._z = e._z;
    this._order = e._order;
    this._onChange?.();
    return this;
  }
  clone(): Euler {
    return new Euler(this._x, this._y, this._z, this._order);
  }
}

export class Quaternion {
  private _x: number;
  private _y: number;
  private _z: number;
  private _w: number;
  _onChange: (() => void) | null = null;

  constructor(x = 0, y = 0, z = 0, w = 1) {
    this._x = x;
    this._y = y;
    this._z = z;
    this._w = w;
  }
  get x() { return this._x; }
  set x(v: number) { this._x = v; this._onChange?.(); }
  get y() { return this._y; }
  set y(v: number) { this._y = v; this._onChange?.(); }
  get z() { return this._z; }
  set z(v: number) { this._z = v; this._onChange?.(); }
  get w() { return this._w; }
  set w(v: number) { this._w = v; this._onChange?.(); }

  set(x: number, y: number, z: number, w: number): this {
    this._x = x;
    this._y = y;
    this._z = z;
    this._w = w;
    this._onChange?.();
    return this;
  }
  identity(): this {
    return this.set(0, 0, 0, 1);
  }
  copy(q: Quaternion): this {
    this._x = q._x;
    this._y = q._y;
    this._z = q._z;
    this._w = q._w;
    this._onChange?.();
    return this;
  }
  clone(): Quaternion {
    return new Quaternion(this._x, this._y, this._z, this._w);
  }
  setFromEuler(euler: Euler, notify = true): this {
    const hx = euler.x * 0.5, hy = euler.y * 0.5, hz = euler.z * 0.5;
    const cx = Math.cos(hx), sx = Math.sin(hx);
    const cy = Math.cos(hy), sy = Math.sin(hy);
    const cz = Math.cos(hz), sz = Math.sin(hz);
    switch (euler.order) {
      case 'XYZ':
        this._x = sx * cy * cz + cx * sy * sz;
        this._y = cx * sy * cz - sx * cy * sz;
        this._z = cx * cy * sz + sx * sy * cz;
        this._w = cx * cy * cz - sx * sy * sz;
        break;
      case 'YXZ':
        this._x = sx * cy * cz + cx * sy * sz;
        this._y = cx * sy * cz - sx * cy * sz;
        this._z = cx * cy * sz - sx * sy * cz;
        this._w = cx * cy * cz + sx * sy * sz;
        break;
      case 'ZXY':
        this._x = sx * cy * cz - cx * sy * sz;
        this._y = cx * sy * cz + sx * cy * sz;
        this._z = cx * cy * sz + sx * sy * cz;
        this._w = cx * cy * cz - sx * sy * sz;
        break;
      case 'ZYX':
        this._x = sx * cy * cz - cx * sy * sz;
        this._y = cx * sy * cz + sx * cy * sz;
        this._z = cx * cy * sz - sx * sy * cz;
        this._w = cx * cy * cz + sx * sy * sz;
        break;
      case 'YZX':
        this._x = sx * cy * cz + cx * sy * sz;
        this._y = cx * sy * cz + sx * cy * sz;
        this._z = cx * cy * sz - sx * sy * cz;
        this._w = cx * cy * cz - sx * sy * sz;
        break;
      case 'XZY':
        this._x = sx * cy * cz - cx * sy * sz;
        this._y = cx * sy * cz - sx * cy * sz;
        this._z = cx * cy * sz + sx * sy * cz;
        this._w = cx * cy * cz + sx * sy * sz;
        break;
    }
    if (notify) this._onChange?.();
    return this;
  }
  setFromAxisAngle(axis: Vector3, angle: number): this {
    const h = angle * 0.5, s = Math.sin(h);
    this._x = axis.x * s;
    this._y = axis.y * s;
    this._z = axis.z * s;
    this._w = Math.cos(h);
    this._onChange?.();
    return this;
  }
  setFromUnitVectors(from: Vector3, to: Vector3): this {
    let r = from.dot(to) + 1;
    if (r < 1e-6) {
      r = 0;
      if (Math.abs(from.x) > Math.abs(from.z)) {
        this._x = -from.y; this._y = from.x; this._z = 0; this._w = 0;
      } else {
        this._x = 0; this._y = -from.z; this._z = from.y; this._w = 0;
      }
    } else {
      this._x = from.y * to.z - from.z * to.y;
      this._y = from.z * to.x - from.x * to.z;
      this._z = from.x * to.y - from.y * to.x;
      this._w = r;
    }
    return this.normalize();
  }
  setFromRotationMatrix(m: Matrix4, notify = true): this {
    const e = m.elements;
    const m11 = e[0], m12 = e[4], m13 = e[8];
    const m21 = e[1], m22 = e[5], m23 = e[9];
    const m31 = e[2], m32 = e[6], m33 = e[10];
    const tr = m11 + m22 + m33;
    if (tr > 0) {
      const s = 0.5 / Math.sqrt(tr + 1);
      this._w = 0.25 / s;
      this._x = (m32 - m23) * s;
      this._y = (m13 - m31) * s;
      this._z = (m21 - m12) * s;
    } else if (m11 > m22 && m11 > m33) {
      const s = 2 * Math.sqrt(1 + m11 - m22 - m33);
      this._w = (m32 - m23) / s;
      this._x = 0.25 * s;
      this._y = (m12 + m21) / s;
      this._z = (m13 + m31) / s;
    } else if (m22 > m33) {
      const s = 2 * Math.sqrt(1 + m22 - m11 - m33);
      this._w = (m13 - m31) / s;
      this._x = (m12 + m21) / s;
      this._y = 0.25 * s;
      this._z = (m23 + m32) / s;
    } else {
      const s = 2 * Math.sqrt(1 + m33 - m11 - m22);
      this._w = (m21 - m12) / s;
      this._x = (m13 + m31) / s;
      this._y = (m23 + m32) / s;
      this._z = 0.25 * s;
    }
    if (notify) this._onChange?.();
    return this;
  }
  normalize(): this {
    const l = Math.hypot(this._x, this._y, this._z, this._w);
    if (l === 0) {
      this._x = 0; this._y = 0; this._z = 0; this._w = 1;
    } else {
      const inv = 1 / l;
      this._x *= inv; this._y *= inv; this._z *= inv; this._w *= inv;
    }
    this._onChange?.();
    return this;
  }
}

export class Matrix3 {
  readonly elements = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);

  getNormalMatrix(m4: Matrix4): this {
    const s = m4.elements;
    const d = this.elements;
    const a00 = s[0], a01 = s[1], a02 = s[2];
    const a10 = s[4], a11 = s[5], a12 = s[6];
    const a20 = s[8], a21 = s[9], a22 = s[10];
    const b01 = a22 * a11 - a12 * a21;
    const b11 = -a22 * a10 + a12 * a20;
    const b21 = a21 * a10 - a11 * a20;
    const det = a00 * b01 + a01 * b11 + a02 * b21;
    if (Math.abs(det) < 1e-12) {
      d[0] = 1; d[1] = 0; d[2] = 0;
      d[3] = 0; d[4] = 1; d[5] = 0;
      d[6] = 0; d[7] = 0; d[8] = 1;
      return this;
    }
    const inv = 1 / det;
    // Inverse transpose of upper-left 3x3
    d[0] = b01 * inv;
    d[1] = b11 * inv;
    d[2] = b21 * inv;
    d[3] = (-a22 * a01 + a02 * a21) * inv;
    d[4] = (a22 * a00 - a02 * a20) * inv;
    d[5] = (-a21 * a00 + a01 * a20) * inv;
    d[6] = (a12 * a01 - a02 * a11) * inv;
    d[7] = (-a12 * a00 + a02 * a10) * inv;
    d[8] = (a11 * a00 - a01 * a10) * inv;
    return this;
  }
}

const _tmpQ = new Quaternion();

export class Matrix4 {
  // Column-major 4x4 matrix
  readonly elements = new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]);

  identity(): this {
    const e = this.elements;
    e[0] = 1; e[4] = 0; e[8] = 0; e[12] = 0;
    e[1] = 0; e[5] = 1; e[9] = 0; e[13] = 0;
    e[2] = 0; e[6] = 0; e[10] = 1; e[14] = 0;
    e[3] = 0; e[7] = 0; e[11] = 0; e[15] = 1;
    return this;
  }

  copy(m: Matrix4): this {
    this.elements.set(m.elements);
    return this;
  }

  clone(): Matrix4 {
    return new Matrix4().copy(this);
  }

  setPosition(x: number | Vector3, y?: number, z?: number): this {
    const e = this.elements;
    if (x instanceof Vector3) {
      e[12] = x.x; e[13] = x.y; e[14] = x.z;
    } else {
      e[12] = x; e[13] = y!; e[14] = z!;
    }
    return this;
  }

  makeTranslation(x: number, y: number, z: number): this {
    this.identity();
    this.elements[12] = x;
    this.elements[13] = y;
    this.elements[14] = z;
    return this;
  }

  makeScale(x: number, y: number, z: number): this {
    this.identity();
    this.elements[0] = x;
    this.elements[5] = y;
    this.elements[10] = z;
    return this;
  }

  makeRotationX(theta: number): this {
    const c = Math.cos(theta), s = Math.sin(theta);
    this.identity();
    const e = this.elements;
    e[5] = c; e[9] = -s;
    e[6] = s; e[10] = c;
    return this;
  }

  makeRotationY(theta: number): this {
    const c = Math.cos(theta), s = Math.sin(theta);
    this.identity();
    const e = this.elements;
    e[0] = c; e[8] = s;
    e[2] = -s; e[10] = c;
    return this;
  }

  makeRotationZ(theta: number): this {
    const c = Math.cos(theta), s = Math.sin(theta);
    this.identity();
    const e = this.elements;
    e[0] = c; e[4] = -s;
    e[1] = s; e[5] = c;
    return this;
  }

  makeRotationFromQuaternion(q: Quaternion): this {
    const e = this.elements;
    const x = q.x, y = q.y, z = q.z, w = q.w;
    const x2 = x + x, y2 = y + y, z2 = z + z;
    const xx = x * x2, xy = x * y2, xz = x * z2;
    const yy = y * y2, yz = y * z2, zz = z * z2;
    const wx = w * x2, wy = w * y2, wz = w * z2;
    e[0] = 1 - (yy + zz); e[4] = xy - wz;       e[8] = xz + wy;        e[12] = 0;
    e[1] = xy + wz;       e[5] = 1 - (xx + zz); e[9] = yz - wx;        e[13] = 0;
    e[2] = xz - wy;       e[6] = yz + wx;       e[10] = 1 - (xx + yy); e[14] = 0;
    e[3] = 0;             e[7] = 0;             e[11] = 0;             e[15] = 1;
    return this;
  }

  makeRotationFromEuler(euler: Euler): this {
    _tmpQ.setFromEuler(euler, false);
    return this.makeRotationFromQuaternion(_tmpQ);
  }

  compose(pos: Vector3, quat: Quaternion, scale: Vector3): this {
    const e = this.elements;
    const x = quat.x, y = quat.y, z = quat.z, w = quat.w;
    const x2 = x + x, y2 = y + y, z2 = z + z;
    const xx = x * x2, xy = x * y2, xz = x * z2;
    const yy = y * y2, yz = y * z2, zz = z * z2;
    const wx = w * x2, wy = w * y2, wz = w * z2;
    const sx = scale.x, sy = scale.y, sz = scale.z;
    e[0] = (1 - (yy + zz)) * sx;
    e[1] = (xy + wz) * sx;
    e[2] = (xz - wy) * sx;
    e[3] = 0;
    e[4] = (xy - wz) * sy;
    e[5] = (1 - (xx + zz)) * sy;
    e[6] = (yz + wx) * sy;
    e[7] = 0;
    e[8] = (xz + wy) * sz;
    e[9] = (yz - wx) * sz;
    e[10] = (1 - (xx + yy)) * sz;
    e[11] = 0;
    e[12] = pos.x;
    e[13] = pos.y;
    e[14] = pos.z;
    e[15] = 1;
    return this;
  }

  multiply(m: Matrix4): this {
    return this.multiplyMatrices(this, m);
  }

  premultiply(m: Matrix4): this {
    return this.multiplyMatrices(m, this);
  }

  multiplyMatrices(a: Matrix4, b: Matrix4): this {
    return multiplyMat4Arrays(this.elements, a.elements, b.elements), this;
  }

  invert(): this {
    invertMat4Array(this.elements, this.elements);
    return this;
  }

  makePerspective(fovDeg: number, aspect: number, near: number, far: number): this {
    const f = 1 / Math.tan((fovDeg * Math.PI) / 360);
    const nf = 1 / (near - far);
    const e = this.elements;
    e[0] = f / aspect; e[4] = 0; e[8] = 0; e[12] = 0;
    e[1] = 0; e[5] = f; e[9] = 0; e[13] = 0;
    e[2] = 0; e[6] = 0; e[10] = (far + near) * nf; e[14] = 2 * far * near * nf;
    e[3] = 0; e[7] = 0; e[11] = -1; e[15] = 0;
    return this;
  }

  makeOrthographic(left: number, right: number, top: number, bottom: number, near: number, far: number): this {
    const w = 1 / (right - left);
    const h = 1 / (top - bottom);
    const p = 1 / (far - near);
    const e = this.elements;
    e[0] = 2 * w; e[4] = 0; e[8] = 0; e[12] = -(right + left) * w;
    e[1] = 0; e[5] = 2 * h; e[9] = 0; e[13] = -(top + bottom) * h;
    e[2] = 0; e[6] = 0; e[10] = -2 * p; e[14] = -(far + near) * p;
    e[3] = 0; e[7] = 0; e[11] = 0; e[15] = 1;
    return this;
  }

  lookAt(eye: Vector3, target: Vector3, up: Vector3): this {
    // Camera orientation basis (right, up, back = eye - target)
    let zx = eye.x - target.x, zy = eye.y - target.y, zz = eye.z - target.z;
    let zl = Math.hypot(zx, zy, zz);
    if (zl === 0) { zz = 1; zl = 1; }
    zx /= zl; zy /= zl; zz /= zl;
    let xx = up.y * zz - up.z * zy;
    let xy = up.z * zx - up.x * zz;
    let xz = up.x * zy - up.y * zx;
    let xl = Math.hypot(xx, xy, xz);
    if (xl === 0) {
      zx += 1e-4;
      zl = Math.hypot(zx, zy, zz) || 1;
      zx /= zl; zy /= zl; zz /= zl;
      xx = up.y * zz - up.z * zy;
      xy = up.z * zx - up.x * zz;
      xz = up.x * zy - up.y * zx;
      xl = Math.hypot(xx, xy, xz) || 1;
    }
    xx /= xl; xy /= xl; xz /= xl;
    const yx = zy * xz - zz * xy;
    const yy = zz * xx - zx * xz;
    const yz = zx * xy - zy * xx;
    const e = this.elements;
    e[0] = xx; e[4] = yx; e[8] = zx; e[12] = 0;
    e[1] = xy; e[5] = yy; e[9] = zy; e[13] = 0;
    e[2] = xz; e[6] = yz; e[10] = zz; e[14] = 0;
    e[3] = 0;  e[7] = 0;  e[11] = 0; e[15] = 1;
    return this;
  }
}

export function multiplyMat4Arrays(out: Float32Array, a: Float32Array, b: Float32Array): void {
  multiplyMat4Into(out, 0, a, b);
}

/** Zero-allocation mat4 multiplication directly into `out` at offset `o`. */
export function multiplyMat4Into(out: Float32Array, o: number, a: Float32Array, b: Float32Array): void {
  const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
  const a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
  const a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
  const a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];

  let b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
  out[o + 0] = a00 * b0 + a10 * b1 + a20 * b2 + a30 * b3;
  out[o + 1] = a01 * b0 + a11 * b1 + a21 * b2 + a31 * b3;
  out[o + 2] = a02 * b0 + a12 * b1 + a22 * b2 + a32 * b3;
  out[o + 3] = a03 * b0 + a13 * b1 + a23 * b2 + a33 * b3;

  b0 = b[4]; b1 = b[5]; b2 = b[6]; b3 = b[7];
  out[o + 4] = a00 * b0 + a10 * b1 + a20 * b2 + a30 * b3;
  out[o + 5] = a01 * b0 + a11 * b1 + a21 * b2 + a31 * b3;
  out[o + 6] = a02 * b0 + a12 * b1 + a22 * b2 + a32 * b3;
  out[o + 7] = a03 * b0 + a13 * b1 + a23 * b2 + a33 * b3;

  b0 = b[8]; b1 = b[9]; b2 = b[10]; b3 = b[11];
  out[o + 8] = a00 * b0 + a10 * b1 + a20 * b2 + a30 * b3;
  out[o + 9] = a01 * b0 + a11 * b1 + a21 * b2 + a31 * b3;
  out[o + 10] = a02 * b0 + a12 * b1 + a22 * b2 + a32 * b3;
  out[o + 11] = a03 * b0 + a13 * b1 + a23 * b2 + a33 * b3;

  b0 = b[12]; b1 = b[13]; b2 = b[14]; b3 = b[15];
  out[o + 12] = a00 * b0 + a10 * b1 + a20 * b2 + a30 * b3;
  out[o + 13] = a01 * b0 + a11 * b1 + a21 * b2 + a31 * b3;
  out[o + 14] = a02 * b0 + a12 * b1 + a22 * b2 + a32 * b3;
  out[o + 15] = a03 * b0 + a13 * b1 + a23 * b2 + a33 * b3;
}

export function invertMat4Array(out: Float32Array, m: Float32Array): void {
  const n11 = m[0], n21 = m[1], n31 = m[2], n41 = m[3];
  const n12 = m[4], n22 = m[5], n32 = m[6], n42 = m[7];
  const n13 = m[8], n23 = m[9], n33 = m[10], n43 = m[11];
  const n14 = m[12], n24 = m[13], n34 = m[14], n44 = m[15];

  const t11 = n23 * n34 * n42 - n24 * n33 * n42 + n24 * n32 * n43 - n22 * n34 * n43 - n23 * n32 * n44 + n22 * n33 * n44;
  const t12 = n14 * n33 * n42 - n13 * n34 * n42 - n14 * n32 * n43 + n12 * n34 * n43 + n13 * n32 * n44 - n12 * n33 * n44;
  const t13 = n13 * n24 * n42 - n14 * n23 * n42 + n14 * n22 * n43 - n12 * n24 * n43 - n13 * n22 * n44 + n12 * n23 * n44;
  const t14 = n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 + n12 * n24 * n33 + n13 * n22 * n34 - n12 * n23 * n34;

  const det = n11 * t11 + n21 * t12 + n31 * t13 + n41 * t14;
  if (Math.abs(det) < 1e-14) {
    out.fill(0);
    out[0] = out[5] = out[10] = out[15] = 1;
    return;
  }
  const inv = 1 / det;
  out[0] = t11 * inv;
  out[1] = (n24 * n33 * n41 - n23 * n34 * n41 - n24 * n31 * n43 + n21 * n34 * n43 + n23 * n31 * n44 - n21 * n33 * n44) * inv;
  out[2] = (n22 * n34 * n41 - n24 * n32 * n41 + n24 * n31 * n42 - n21 * n34 * n42 - n22 * n31 * n44 + n21 * n32 * n44) * inv;
  out[3] = (n23 * n32 * n41 - n22 * n33 * n41 - n23 * n31 * n42 + n21 * n33 * n42 + n22 * n31 * n43 - n21 * n32 * n43) * inv;
  out[4] = t12 * inv;
  out[5] = (n13 * n34 * n41 - n14 * n33 * n41 + n14 * n31 * n43 - n11 * n34 * n43 - n13 * n31 * n44 + n11 * n33 * n44) * inv;
  out[6] = (n14 * n32 * n41 - n12 * n34 * n41 - n14 * n31 * n42 + n11 * n34 * n42 + n12 * n31 * n44 - n11 * n32 * n44) * inv;
  out[7] = (n12 * n33 * n41 - n13 * n32 * n41 + n13 * n31 * n42 - n11 * n33 * n42 - n12 * n31 * n43 + n11 * n32 * n43) * inv;
  out[8] = t13 * inv;
  out[9] = (n14 * n23 * n41 - n13 * n24 * n41 - n14 * n21 * n43 + n11 * n24 * n43 + n13 * n21 * n44 - n11 * n23 * n44) * inv;
  out[10] = (n12 * n24 * n41 - n14 * n22 * n41 + n14 * n21 * n42 - n11 * n24 * n42 - n12 * n21 * n44 + n11 * n22 * n44) * inv;
  out[11] = (n13 * n22 * n41 - n12 * n23 * n41 - n13 * n21 * n42 + n11 * n23 * n42 + n12 * n21 * n43 - n11 * n22 * n43) * inv;
  out[12] = t14 * inv;
  out[13] = (n13 * n24 * n31 - n14 * n23 * n31 + n14 * n21 * n33 - n11 * n24 * n33 - n13 * n21 * n34 + n11 * n23 * n34) * inv;
  out[14] = (n14 * n22 * n31 - n12 * n24 * n31 - n14 * n21 * n32 + n11 * n24 * n32 + n12 * n21 * n34 - n11 * n22 * n34) * inv;
  out[15] = (n12 * n23 * n31 - n13 * n22 * n31 + n13 * n21 * n32 - n11 * n23 * n32 - n12 * n21 * n33 + n11 * n22 * n33) * inv;
}

export class Sphere {
  center: Vector3;
  radius: number;
  constructor(center = new Vector3(), radius = -1) {
    this.center = center;
    this.radius = radius;
  }
  clone(): Sphere {
    return new Sphere(this.center.clone(), this.radius);
  }
  copy(s: Sphere): this {
    this.center.copy(s.center);
    this.radius = s.radius;
    return this;
  }
}

// Fast 6-plane frustum stored in a flat Float32Array(24) for zero-allocation culling
export function extractFrustumPlanes(vp: Float32Array, out: Float32Array): void {
  const m0 = vp[0], m1 = vp[1], m2 = vp[2], m3 = vp[3];
  const m4 = vp[4], m5 = vp[5], m6 = vp[6], m7 = vp[7];
  const m8 = vp[8], m9 = vp[9], m10 = vp[10], m11 = vp[11];
  const m12 = vp[12], m13 = vp[13], m14 = vp[14], m15 = vp[15];
  const setPlane = (i: number, a: number, b: number, c: number, d: number) => {
    const inv = 1 / (Math.hypot(a, b, c) || 1);
    out[i] = a * inv; out[i + 1] = b * inv; out[i + 2] = c * inv; out[i + 3] = d * inv;
  };
  setPlane(0, m3 - m0, m7 - m4, m11 - m8, m15 - m12); // right
  setPlane(4, m3 + m0, m7 + m4, m11 + m8, m15 + m12); // left
  setPlane(8, m3 - m1, m7 - m5, m11 - m9, m15 - m13); // top
  setPlane(12, m3 + m1, m7 + m5, m11 + m9, m15 + m13); // bottom
  setPlane(16, m3 - m2, m7 - m6, m11 - m10, m15 - m14); // far
  setPlane(20, m3 + m2, m7 + m6, m11 + m10, m15 + m14); // near
}

export function sphereIntersectsFrustum(planes: Float32Array, cx: number, cy: number, cz: number, r: number): boolean {
  for (let i = 0; i < 24; i += 4) {
    if (planes[i] * cx + planes[i + 1] * cy + planes[i + 2] * cz + planes[i + 3] < -r) return false;
  }
  return true;
}
