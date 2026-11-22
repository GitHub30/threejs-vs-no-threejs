// Zero-dependency shared OrbitCamera with exact WebGPU [0,1] & WebGL [-1,1] projection,
// Gribb-Hartmann frustum extraction, 3D pointer ray projection, and Three.js camera sync.

export class OrbitCamera {
  constructor() {
    this.fovDeg = 50;
    this.near = 0.5;
    this.far = 650.0;
    this.aspect = 1.6;

    // Spherical orbit coordinates around sanctuary center (cinematic low-horizon elevation)
    this.theta = 0.54;
    this.phi = 1.49;
    this.radius = 84.0;

    this.targetTheta = this.theta;
    this.targetPhi = this.phi;
    this.targetRadius = this.radius;

    this.target = [0, 2.0, 0];
    this.position = new Float32Array(3);

    this.autoOrbit = true;
    this.autoOrbitSpeed = 0.14;

    this.viewMatrix = new Float32Array(16);
    this.projWebGPU = new Float32Array(16);
    this.projWebGL = new Float32Array(16);
    this.viewProjWebGPU = new Float32Array(16);
    this.viewProjWebGL = new Float32Array(16);
    this.invViewProjWebGPU = new Float32Array(16);
    this.frustumWebGPU = new Float32Array(24);
    this.frustumWebGL = new Float32Array(24);

    // Interactive pointer vortex state
    this.pointerActive = false;
    this.pointerAttract = false;
    this.pointerNdcX = 0;
    this.pointerNdcY = 0;
    this.pointerWorld = new Float32Array([0, 0, 0]);
    this.pointerMode = 0; // 0 = off, 1 = attract, -1 = repel

    this._dragging = false;
    this._lastX = 0;
    this._lastY = 0;
    this._pinchDist = 0;

    this.updateMatrices();
  }

  attachElement(el, onInteract) {
    const onPointerDown = (e) => {
      if (e.button === 0 && !e.shiftKey) {
        this._dragging = true;
        this._lastX = e.clientX;
        this._lastY = e.clientY;
        el.setPointerCapture?.(e.pointerId);
      } else if (e.button === 2 || e.shiftKey) {
        this.pointerAttract = true;
      }
      if (onInteract) onInteract();
    };

    const onPointerMove = (e) => {
      const rect = el.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        this.pointerNdcX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        this.pointerNdcY = -(((e.clientY - rect.top) / rect.height) * 2 - 1);
        this.pointerActive = true;
      }

      if (this._dragging) {
        const dx = e.clientX - this._lastX;
        const dy = e.clientY - this._lastY;
        this._lastX = e.clientX;
        this._lastY = e.clientY;
        this.targetTheta -= dx * 0.0065;
        this.targetPhi = Math.max(0.22, Math.min(1.52, this.targetPhi - dy * 0.0055));
        this.autoOrbit = false;
      }
    };

    const onPointerUp = (e) => {
      this._dragging = false;
      this.pointerAttract = false;
      try { el.releasePointerCapture?.(e.pointerId); } catch {}
    };

    const onPointerLeave = () => {
      this._dragging = false;
      this.pointerActive = false;
      this.pointerAttract = false;
    };

    const onWheel = (e) => {
      e.preventDefault();
      const zoomFactor = 1.0 + Math.sign(e.deltaY) * 0.09;
      this.targetRadius = Math.max(36, Math.min(185, this.targetRadius * zoomFactor));
    };

    const onContextMenu = (e) => e.preventDefault();

    el.addEventListener('pointerdown', onPointerDown);
    el.addEventListener('pointermove', onPointerMove);
    el.addEventListener('pointerup', onPointerUp);
    el.addEventListener('pointercancel', onPointerUp);
    el.addEventListener('pointerleave', onPointerLeave);
    el.addEventListener('wheel', onWheel, { passive: false });
    el.addEventListener('contextmenu', onContextMenu);

    return () => {
      el.removeEventListener('pointerdown', onPointerDown);
      el.removeEventListener('pointermove', onPointerMove);
      el.removeEventListener('pointerup', onPointerUp);
      el.removeEventListener('pointercancel', onPointerUp);
      el.removeEventListener('pointerleave', onPointerLeave);
      el.removeEventListener('wheel', onWheel);
      el.removeEventListener('contextmenu', onContextMenu);
    };
  }

  resetForBenchmark() {
    this.theta = 0.54;
    this.phi = 1.49;
    this.radius = 84.0;
    this.targetTheta = this.theta;
    this.targetPhi = this.phi;
    this.targetRadius = this.radius;
    this.autoOrbit = true;
    this.pointerActive = false;
    this.pointerAttract = false;
    this.pointerMode = 0;
    this.updateMatrices();
  }

  setAspect(aspect) {
    if (aspect > 0 && Number.isFinite(aspect)) {
      this.aspect = aspect;
    }
  }

  update(dt) {
    const safeDt = Math.min(0.05, Math.max(0.001, dt));
    if (this.autoOrbit) {
      this.targetTheta += this.autoOrbitSpeed * safeDt;
    }
    const damp = 1.0 - Math.exp(-8.5 * safeDt);
    this.theta += (this.targetTheta - this.theta) * damp;
    this.phi += (this.targetPhi - this.phi) * damp;
    this.radius += (this.targetRadius - this.radius) * damp;

    this.updateMatrices();
    this.updatePointerWorld();
  }

  updateMatrices() {
    const sinPhi = Math.sin(this.phi);
    const cosPhi = Math.cos(this.phi);
    const sinTheta = Math.sin(this.theta);
    const cosTheta = Math.cos(this.theta);

    const px = this.target[0] + this.radius * sinPhi * sinTheta;
    const py = this.target[1] + this.radius * cosPhi;
    const pz = this.target[2] + this.radius * sinPhi * cosTheta;

    this.position[0] = px;
    this.position[1] = py;
    this.position[2] = pz;

    lookAt(this.viewMatrix, px, py, pz, this.target[0], this.target[1], this.target[2], 0, 1, 0);

    const fovRad = (this.fovDeg * Math.PI) / 180;
    perspectiveWebGPU(this.projWebGPU, fovRad, this.aspect, this.near, this.far);
    perspectiveWebGL(this.projWebGL, fovRad, this.aspect, this.near, this.far);

    mulMat4(this.viewProjWebGPU, this.projWebGPU, this.viewMatrix);
    mulMat4(this.viewProjWebGL, this.projWebGL, this.viewMatrix);

    invertMat4(this.invViewProjWebGPU, this.viewProjWebGPU);
    extractFrustumPlanes(this.frustumWebGPU, this.viewProjWebGPU, true);
    extractFrustumPlanes(this.frustumWebGL, this.viewProjWebGL, false);
  }

  updatePointerWorld() {
    if (!this.pointerActive || this._dragging) {
      this.pointerMode = 0;
      return;
    }
    // Unproject pointer ray from near (z=0 in WebGPU NDC) to far (z=1)
    const rayOrigin = this.position;
    const farPt = unprojectPoint(this.invViewProjWebGPU, this.pointerNdcX, this.pointerNdcY, 1.0);
    let dx = farPt[0] - rayOrigin[0];
    let dy = farPt[1] - rayOrigin[1];
    let dz = farPt[2] - rayOrigin[2];
    const len = Math.hypot(dx, dy, dz) || 1.0;
    dx /= len; dy /= len; dz /= len;

    // Intersect ray with sphere/plane passing through arena center perpendicular to camera view
    const toCenterX = this.target[0] - rayOrigin[0];
    const toCenterY = this.target[1] - rayOrigin[1];
    const toCenterZ = this.target[2] - rayOrigin[2];
    const t = Math.max(15, Math.min(180, toCenterX * dx + toCenterY * dy + toCenterZ * dz));

    this.pointerWorld[0] = rayOrigin[0] + dx * t;
    this.pointerWorld[1] = Math.max(-28, Math.min(28, rayOrigin[1] + dy * t));
    this.pointerWorld[2] = rayOrigin[2] + dz * t;
    this.pointerMode = this.pointerAttract ? 1 : -1;
  }

  syncThreeCamera(threeCam) {
    threeCam.fov = this.fovDeg;
    threeCam.near = this.near;
    threeCam.far = this.far;
    if (Math.abs(threeCam.aspect - this.aspect) > 1e-5) {
      threeCam.aspect = this.aspect;
      threeCam.updateProjectionMatrix();
    }
    threeCam.position.set(this.position[0], this.position[1], this.position[2]);
    threeCam.lookAt(this.target[0], this.target[1], this.target[2]);
    threeCam.updateMatrixWorld();
  }
}

// Column-major 4x4 matrix math (matching WebGPU WGSL & OpenGL/Three.js conventions)

export function lookAt(out, ex, ey, ez, tx, ty, tz, ux, uy, uz) {
  let zx = ex - tx, zy = ey - ty, zz = ez - tz;
  let len = Math.hypot(zx, zy, zz) || 1;
  zx /= len; zy /= len; zz /= len;

  let xx = uy * zz - uz * zy;
  let xy = uz * zx - ux * zz;
  let xz = ux * zy - uy * zx;
  len = Math.hypot(xx, xy, xz) || 1;
  xx /= len; xy /= len; xz /= len;

  const yx = zy * xz - zz * xy;
  const yy = zz * xx - zx * xz;
  const yz = zx * xy - zy * xx;

  out[0] = xx; out[1] = yx; out[2] = zx; out[3] = 0;
  out[4] = xy; out[5] = yy; out[6] = zy; out[7] = 0;
  out[8] = xz; out[9] = yz; out[10] = zz; out[11] = 0;
  out[12] = -(xx * ex + xy * ey + xz * ez);
  out[13] = -(yx * ex + yy * ey + yz * ez);
  out[14] = -(zx * ex + zy * ey + zz * ez);
  out[15] = 1;
}

export function perspectiveWebGPU(out, fovy, aspect, near, far) {
  const f = 1.0 / Math.tan(fovy * 0.5);
  const nf = 1.0 / (near - far);
  out.fill(0);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = far * nf;
  out[11] = -1;
  out[14] = far * near * nf;
}

export function perspectiveWebGL(out, fovy, aspect, near, far) {
  const f = 1.0 / Math.tan(fovy * 0.5);
  const nf = 1.0 / (near - far);
  out.fill(0);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = (far + near) * nf;
  out[11] = -1;
  out[14] = 2 * far * near * nf;
}

export function mulMat4(out, a, b) {
  for (let col = 0; col < 4; col++) {
    const b0 = b[col * 4], b1 = b[col * 4 + 1], b2 = b[col * 4 + 2], b3 = b[col * 4 + 3];
    out[col * 4]     = a[0] * b0 + a[4] * b1 + a[8]  * b2 + a[12] * b3;
    out[col * 4 + 1] = a[1] * b0 + a[5] * b1 + a[9]  * b2 + a[13] * b3;
    out[col * 4 + 2] = a[2] * b0 + a[6] * b1 + a[10] * b2 + a[14] * b3;
    out[col * 4 + 3] = a[3] * b0 + a[7] * b1 + a[11] * b2 + a[15] * b3;
  }
}

export function extractFrustumPlanes(out, m, isWebGPU) {
  // Column-major matrix rows: r0=(m0,m4,m8,m12), r1=(m1,m5,m9,m13), r2=(m2,m6,m10,m14), r3=(m3,m7,m11,m15)
  const setPlane = (idx, a, b, c, d) => {
    const inv = 1.0 / (Math.hypot(a, b, c) || 1.0);
    out[idx * 4]     = a * inv;
    out[idx * 4 + 1] = b * inv;
    out[idx * 4 + 2] = c * inv;
    out[idx * 4 + 3] = d * inv;
  };
  // Left: r3 + r0
  setPlane(0, m[3] + m[0], m[7] + m[4], m[11] + m[8],  m[15] + m[12]);
  // Right: r3 - r0
  setPlane(1, m[3] - m[0], m[7] - m[4], m[11] - m[8],  m[15] - m[12]);
  // Bottom: r3 + r1
  setPlane(2, m[3] + m[1], m[7] + m[5], m[11] + m[9],  m[15] + m[13]);
  // Top: r3 - r1
  setPlane(3, m[3] - m[1], m[7] - m[5], m[11] - m[9],  m[15] - m[13]);
  // Near: r2 (WebGPU z in [0,1]) or r3 + r2 (WebGL z in [-1,1])
  if (isWebGPU) {
    setPlane(4, m[2], m[6], m[10], m[14]);
  } else {
    setPlane(4, m[3] + m[2], m[7] + m[6], m[11] + m[10], m[15] + m[14]);
  }
  // Far: r3 - r2
  setPlane(5, m[3] - m[2], m[7] - m[6], m[11] - m[10], m[15] - m[14]);
}

export function invertMat4(out, m) {
  const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3];
  const a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
  const a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11];
  const a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];

  const b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10, b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30, b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;

  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!det) return;
  det = 1.0 / det;

  out[0]  = (a11 * b11 - a12 * b10 + a13 * b09) * det;
  out[1]  = (a02 * b10 - a01 * b11 - a03 * b09) * det;
  out[2]  = (a31 * b05 - a32 * b04 + a33 * b03) * det;
  out[3]  = (a22 * b04 - a21 * b05 - a23 * b03) * det;
  out[4]  = (a12 * b08 - a10 * b11 - a13 * b07) * det;
  out[5]  = (a00 * b11 - a02 * b08 + a03 * b07) * det;
  out[6]  = (a32 * b02 - a30 * b05 - a33 * b01) * det;
  out[7]  = (a20 * b05 - a22 * b02 + a23 * b01) * det;
  out[8]  = (a10 * b10 - a11 * b08 + a13 * b06) * det;
  out[9]  = (a01 * b08 - a00 * b10 - a03 * b06) * det;
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
  out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
  out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
  out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
}

function unprojectPoint(invVP, x, y, z) {
  const w = invVP[3] * x + invVP[7] * y + invVP[11] * z + invVP[15] || 1.0;
  return [
    (invVP[0] * x + invVP[4] * y + invVP[8]  * z + invVP[12]) / w,
    (invVP[1] * x + invVP[5] * y + invVP[9]  * z + invVP[13]) / w,
    (invVP[2] * x + invVP[6] * y + invVP[10] * z + invVP[14]) / w,
  ];
}
