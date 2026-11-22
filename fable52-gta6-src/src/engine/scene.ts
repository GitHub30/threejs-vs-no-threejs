import { BufferGeometry } from './geometry';
import { Material } from './materials';
import {
  Color,
  ColorLike,
  Euler,
  Matrix4,
  Quaternion,
  Sphere,
  Vector2,
  Vector3,
  invertMat4Array,
  multiplyMat4Arrays,
  multiplyMat4Into,
} from './math';

let nextObjId = 1;
const _up = new Vector3(0, 1, 0);
const _lookMat = new Matrix4();
const _tmpMat = new Float32Array(16);

export class Object3D {
  static readonly DEFAULT_UP = new Vector3(0, 1, 0);
  readonly id = nextObjId++;
  name = '';
  parent: Object3D | null = null;
  children: Object3D[] = [];
  position = new Vector3();
  rotation = new Euler();
  quaternion = new Quaternion();
  scale = new Vector3(1, 1, 1);
  matrix = new Matrix4();
  matrixWorld = new Matrix4();
  visible = true;
  castShadow = false;
  receiveShadow = false;
  frustumCulled = true;
  renderOrder = 0;
  userData: Record<string, any> = {};

  // Fast dirty tracking so static world subtrees skip compose() and multiplyMatrices() every frame
  private _px = NaN; private _py = NaN; private _pz = NaN;
  private _qx = 0; private _qy = 0; private _qz = 0; private _qw = 1;
  private _sx = 1; private _sy = 1; private _sz = 1;
  private _rotDirty = true;
  _worldDirty = true;
  _sceneRoot: Scene | null = null;

  constructor() {
    this.rotation._onChange = () => {
      this.quaternion.setFromEuler(this.rotation, false);
      this._rotDirty = true;
    };
    this.quaternion._onChange = () => {
      this._rotDirty = true;
    };
  }

  protected copyBase(src: Object3D, recursive: boolean): this {
    this.name = src.name;
    this.position.copy(src.position);
    this.rotation.copy(src.rotation);
    this.quaternion.copy(src.quaternion);
    this.scale.copy(src.scale);
    this.visible = src.visible;
    this.castShadow = src.castShadow;
    this.receiveShadow = src.receiveShadow;
    this.frustumCulled = src.frustumCulled;
    this.renderOrder = src.renderOrder;
    this.userData = { ...src.userData };
    if (recursive) {
      for (const ch of src.children) this.add(ch.clone(true));
    }
    return this;
  }

  clone(recursive = true): Object3D {
    return new Object3D().copyBase(this, recursive);
  }

  add(...objects: Object3D[]): this {
    for (const obj of objects) {
      if (!obj || obj === this) continue;
      if (obj.parent) obj.parent.remove(obj);
      obj.parent = this;
      this.children.push(obj);
      obj._worldDirty = true;
      const root = this._sceneRoot ?? (this instanceof Scene ? this : null);
      if (root) {
        obj._propagateSceneRoot(root);
        root._structureVersion++;
      }
    }
    return this;
  }

  remove(...objects: Object3D[]): this {
    for (const obj of objects) {
      const idx = this.children.indexOf(obj);
      if (idx >= 0) {
        this.children.splice(idx, 1);
        obj.parent = null;
        const root = obj._sceneRoot;
        if (root) {
          obj._propagateSceneRoot(null);
          root._structureVersion++;
        }
      }
    }
    return this;
  }

  _propagateSceneRoot(root: Scene | null): void {
    this._sceneRoot = root;
    for (let i = 0; i < this.children.length; i++) {
      this.children[i]._propagateSceneRoot(root);
    }
  }

  traverse(fn: (obj: Object3D) => void): void {
    fn(this);
    const ch = this.children;
    for (let i = 0; i < ch.length; i++) ch[i].traverse(fn);
  }

  lookAt(target: Vector3 | number, y?: number, z?: number): void {
    const t = target instanceof Vector3 ? target : new Vector3(target, y!, z!);
    // Cameras look down -Z (eye -> target), regular objects face +Z
    if (this instanceof Camera) {
      _lookMat.lookAt(this.position, t, _up);
    } else {
      _lookMat.lookAt(t, this.position, _up);
    }
    this.quaternion.setFromRotationMatrix(_lookMat, true);
  }

  updateMatrix(): boolean {
    const p = this.position, q = this.quaternion, s = this.scale;
    if (
      this._rotDirty ||
      p.x !== this._px || p.y !== this._py || p.z !== this._pz ||
      s.x !== this._sx || s.y !== this._sy || s.z !== this._sz ||
      q.x !== this._qx || q.y !== this._qy || q.z !== this._qz || q.w !== this._qw
    ) {
      this.matrix.compose(p, q, s);
      this._px = p.x; this._py = p.y; this._pz = p.z;
      this._qx = q.x; this._qy = q.y; this._qz = q.z; this._qw = q.w;
      this._sx = s.x; this._sy = s.y; this._sz = s.z;
      this._rotDirty = false;
      this._worldDirty = true;
      return true;
    }
    return false;
  }

  updateMatrixWorld(force = false): void {
    const localChanged = this.updateMatrix();
    const dirty = force || localChanged || this._worldDirty;
    if (dirty) {
      if (this.parent === null) {
        this.matrixWorld.elements.set(this.matrix.elements);
      } else {
        multiplyMat4Arrays(this.matrixWorld.elements, this.parent.matrixWorld.elements, this.matrix.elements);
      }
      this._worldDirty = false;
    }
    const ch = this.children;
    for (let i = 0; i < ch.length; i++) {
      ch[i].updateMatrixWorld(dirty);
    }
  }
}

export class Group extends Object3D {
  readonly isGroup = true;
  override clone(recursive = true): Group {
    return new Group().copyBase(this, recursive);
  }
}

export class Fog {
  color: Color;
  near: number;
  far: number;
  constructor(color: ColorLike, near = 1, far = 1000) {
    this.color = new Color(color);
    this.near = near;
    this.far = far;
  }
}

export class Scene extends Object3D {
  readonly isScene = true;
  fog: Fog | null = null;
  environment: unknown = null;
  environmentIntensity = 0.5;
  _structureVersion = 1;

  constructor() {
    super();
    this._sceneRoot = this;
  }
}

export class Camera extends Object3D {
  readonly isCamera = true;
  projectionMatrix = new Matrix4();
  matrixWorldInverse = new Matrix4();
  near = 0.1;
  far = 2000;

  override updateMatrixWorld(force = false): void {
    super.updateMatrixWorld(force);
    invertMat4Array(this.matrixWorldInverse.elements, this.matrixWorld.elements);
  }
}

export class PerspectiveCamera extends Camera {
  fov: number;
  aspect: number;

  constructor(fov = 60, aspect = 1, near = 0.1, far = 2000) {
    super();
    this.fov = fov;
    this.aspect = aspect;
    this.near = near;
    this.far = far;
    this.updateProjectionMatrix();
  }

  updateProjectionMatrix(): void {
    this.projectionMatrix.makePerspective(this.fov, this.aspect, this.near, this.far);
  }
}

export class OrthographicCamera extends Camera {
  left: number;
  right: number;
  top: number;
  bottom: number;

  constructor(left = -1, right = 1, top = 1, bottom = -1, near = 0.1, far = 2000) {
    super();
    this.left = left;
    this.right = right;
    this.top = top;
    this.bottom = bottom;
    this.near = near;
    this.far = far;
    this.updateProjectionMatrix();
  }

  updateProjectionMatrix(): void {
    this.projectionMatrix.makeOrthographic(this.left, this.right, this.top, this.bottom, this.near, this.far);
  }
}

export class Mesh extends Object3D {
  readonly isMesh = true;
  geometry: BufferGeometry;
  material: Material | Material[];
  boundingSphere: Sphere | null = null;

  constructor(geometry: BufferGeometry, material: Material | Material[]) {
    super();
    this.geometry = geometry;
    this.material = material;
  }

  override clone(recursive = true): Mesh {
    const m = new Mesh(this.geometry, this.material).copyBase(this, recursive);
    if (this.boundingSphere) m.boundingSphere = this.boundingSphere.clone();
    return m;
  }
}

export class InstancedMesh extends Mesh {
  readonly isInstancedMesh = true;
  count: number;
  instanceMatrix: { array: Float32Array; version: number; needsUpdate: boolean };
  instanceColor: { array: Float32Array; version: number; needsUpdate: boolean } | null = null;

  constructor(geometry: BufferGeometry, material: Material, count: number) {
    super(geometry, material);
    this.count = count;
    const matArr = new Float32Array(count * 16);
    for (let i = 0; i < count; i++) {
      const o = i * 16;
      matArr[o] = matArr[o + 5] = matArr[o + 10] = matArr[o + 15] = 1;
    }
    const inst = {
      array: matArr,
      version: 1,
      set needsUpdate(v: boolean) { if (v) this.version++; },
      get needsUpdate() { return false; },
    };
    this.instanceMatrix = inst;
    this.frustumCulled = false; // world-wide instance clouds (trees/lamps) span the city
  }

  setMatrixAt(index: number, matrix: Matrix4): void {
    this.instanceMatrix.array.set(matrix.elements, index * 16);
    this.instanceMatrix.version++;
  }

  setColorAt(index: number, color: Color): void {
    if (!this.instanceColor) {
      const colArr = new Float32Array(this.count * 3);
      colArr.fill(1);
      this.instanceColor = {
        array: colArr,
        version: 1,
        set needsUpdate(v: boolean) { if (v) this.version++; },
        get needsUpdate() { return false; },
      };
    }
    const o = index * 3;
    this.instanceColor.array[o] = color.r;
    this.instanceColor.array[o + 1] = color.g;
    this.instanceColor.array[o + 2] = color.b;
    this.instanceColor.version++;
  }
  _onDispose: ((im: InstancedMesh) => void) | null = null;

  dispose(): void {
    this._onDispose?.(this);
  }
}

export class Bone extends Object3D {
  readonly isBone = true;
}

const _tmpSkin2 = new Float32Array(16);

export class Skeleton {
  bones: Bone[];
  boneInverses: Matrix4[];
  boneMatrices: Float32Array;

  constructor(bones: Bone[], boneInverses?: Matrix4[]) {
    this.bones = bones;
    this.boneMatrices = new Float32Array(Math.max(1, bones.length) * 16);
    if (boneInverses && boneInverses.length === bones.length) {
      this.boneInverses = boneInverses;
    } else {
      this.boneInverses = bones.map((b) => {
        const inv = new Matrix4();
        invertMat4Array(inv.elements, b.matrixWorld.elements);
        return inv;
      });
    }
  }

  /** Zero-allocation bone skin matrix update directly into boneMatrices. */
  update(meshWorldInv?: Float32Array, bindMatrix?: Float32Array, bindMatrixInv?: Float32Array): void {
    const bones = this.bones;
    const invs = this.boneInverses;
    const out = this.boneMatrices;
    for (let i = 0; i < bones.length; i++) {
      const off = i * 16;
      const bw = bones[i].matrixWorld.elements;
      const bi = invs[i].elements;
      if (bindMatrixInv) {
        multiplyMat4Arrays(_tmpSkin2, bindMatrixInv, bw);
        multiplyMat4Arrays(_tmpMat, _tmpSkin2, bi);
        multiplyMat4Into(out, off, _tmpMat, bindMatrix!);
      } else if (meshWorldInv) {
        multiplyMat4Arrays(_tmpMat, meshWorldInv, bw);
        multiplyMat4Into(out, off, _tmpMat, bi);
      } else {
        multiplyMat4Into(out, off, bw, bi);
      }
    }
  }
}

export class SkinnedMesh extends Mesh {
  readonly isSkinnedMesh = true;
  skeleton!: Skeleton;
  bindMatrix = new Matrix4();
  bindMatrixInverse = new Matrix4();
  _hasNonIdentityBind = false;

  constructor(geometry: BufferGeometry, material: Material | Material[]) {
    super(geometry, material);
  }

  bind(skeleton: Skeleton, bindMatrix?: Matrix4): void {
    this.skeleton = skeleton;
    if (bindMatrix) {
      this.bindMatrix.copy(bindMatrix);
      const e = bindMatrix.elements;
      this._hasNonIdentityBind =
        e[0] !== 1 || e[5] !== 1 || e[10] !== 1 || e[15] !== 1 ||
        e[1] !== 0 || e[2] !== 0 || e[3] !== 0 || e[4] !== 0 ||
        e[6] !== 0 || e[7] !== 0 || e[8] !== 0 || e[9] !== 0 ||
        e[11] !== 0 || e[12] !== 0 || e[13] !== 0 || e[14] !== 0;
    } else {
      this.bindMatrix.identity();
      this._hasNonIdentityBind = false;
    }
    invertMat4Array(this.bindMatrixInverse.elements, this.bindMatrix.elements);
  }
}

export class Points extends Object3D {
  readonly isPoints = true;
  geometry: BufferGeometry;
  material: Material;
  constructor(geometry: BufferGeometry, material: Material) {
    super();
    this.geometry = geometry;
    this.material = material;
  }
}

export class LineSegments extends Object3D {
  readonly isLineSegments = true;
  geometry: BufferGeometry;
  material: Material;
  constructor(geometry: BufferGeometry, material: Material) {
    super();
    this.geometry = geometry;
    this.material = material;
  }
}

// =========================================================================
// Lights
// =========================================================================

export class Light extends Object3D {
  readonly isLight = true;
  color: Color;
  intensity: number;
  constructor(color: ColorLike = 0xffffff, intensity = 1) {
    super();
    this.color = new Color(color);
    this.intensity = intensity;
  }
}

export class DirectionalLight extends Light {
  readonly isDirectionalLight = true;
  target = new Object3D();
  shadow = {
    mapSize: new Vector2(1024, 1024),
    camera: new OrthographicCamera(-75, 75, 75, -75, 1, 420),
    bias: -0.0004,
    normalBias: 0.06,
    map: null as { dispose(): void } | null,
  };
  constructor(color: ColorLike = 0xffffff, intensity = 1) {
    super(color, intensity);
  }
}

export class HemisphereLight extends Light {
  readonly isHemisphereLight = true;
  groundColor: Color;
  constructor(skyColor: ColorLike = 0xffffff, groundColor: ColorLike = 0x444444, intensity = 1) {
    super(skyColor, intensity);
    this.groundColor = new Color(groundColor);
  }
}

export class SpotLight extends Light {
  readonly isSpotLight = true;
  target = new Object3D();
  distance: number;
  angle: number;
  penumbra: number;
  decay: number;
  constructor(color: ColorLike = 0xffffff, intensity = 1, distance = 0, angle = Math.PI / 3, penumbra = 0, decay = 2) {
    super(color, intensity);
    this.distance = distance;
    this.angle = angle;
    this.penumbra = penumbra;
    this.decay = decay;
  }
}

export class PointLight extends Light {
  readonly isPointLight = true;
  distance: number;
  decay: number;
  constructor(color: ColorLike = 0xffffff, intensity = 1, distance = 0, decay = 2) {
    super(color, intensity);
    this.distance = distance;
    this.decay = decay;
  }
}
