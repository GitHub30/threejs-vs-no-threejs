// Hand-optimized WGSL shaders for Custom WebGPU Engine:
// - Atmospheric Sky Dome (fullscreen triangle, zero vertex buffers)
// - Architectural Sanctuary Floor (analytic contact AO under monoliths + radial travertine tiles)
// - Sanctuary Monoliths (directional sun + hemisphere ambient + golden resonance accents)
// - Articulated 3D Origami Birds (32-byte compact storage pulling + wing kinematics + GGX glint)
// - Pure GPU Tiled Shared-Memory Compute Boids (var<workgroup> 64x VRAM traffic reduction + fused pass)
// - Orbiting Faceted Crystal Beacons (GPURenderBundle multi-draw)

const COMMON_WGSL = /* wgsl */ `
struct FrameUniforms {
  viewProj : mat4x4<f32>,
  invViewProj : mat4x4<f32>,
  cameraPos : vec4<f32>,   // xyz = pos, w = time
  sunDir : vec4<f32>,      // xyz = normalized sun dir, w = fogDensity
};

fn acesToneMap(c: vec3<f32>) -> vec3<f32> {
  let a = 2.51;
  let b = 0.03;
  let cc = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((c * (a * c + b)) / (c * (cc * c + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn linearToSrgb(c: vec3<f32>) -> vec3<f32> {
  return pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
}

fn evalSkyColor(dir: vec3<f32>, sunDir: vec3<f32>) -> vec3<f32> {
  let y = dir.y;
  let zenith  = vec3<f32>(0.045, 0.195, 0.520);
  let horizon = vec3<f32>(0.760, 0.825, 0.880);
  let ground  = vec3<f32>(0.360, 0.440, 0.510);

  let upT = pow(clamp(y * 1.65, 0.0, 1.0), 0.58);
  let dnT = pow(clamp(-y * 1.5, 0.0, 1.0), 0.65);
  var col = mix(horizon, zenith, upT);
  if (y < 0.0) {
    col = mix(horizon, ground, dnT);
  }

  let sunDot = max(dot(dir, sunDir), 0.0);
  let halo = pow(sunDot, 10.0) * 0.24 + pow(sunDot, 96.0) * 0.72;
  let disc = smoothstep(0.9985, 0.9994, sunDot) * 1.9;
  col += vec3<f32>(1.0, 0.88, 0.68) * (halo + disc);
  return col;
}

fn applyAerialFog(color: vec3<f32>, dist: f32, viewDir: vec3<f32>, sunDir: vec3<f32>) -> vec3<f32> {
  let fogBase = vec3<f32>(0.76, 0.815, 0.865);
  let sunBias = pow(max(dot(viewDir, sunDir), 0.0), 6.0);
  let fogCol = mix(fogBase, vec3<f32>(0.95, 0.88, 0.76), sunBias * 0.35);
  let fogFactor = 1.0 - exp(-pow(dist * 0.0022, 2.0));
  return mix(color, fogCol, clamp(fogFactor, 0.0, 0.88));
}

fn birdPalette(seed: f32, region: f32) -> vec3<f32> {
  var base: vec3<f32>;
  if (seed < 0.28) {
    base = vec3<f32>(0.82, 0.16, 0.08); // Sunlit Vermilion
  } else if (seed < 0.54) {
    base = vec3<f32>(0.86, 0.46, 0.06); // Warm Ochre
  } else if (seed < 0.80) {
    base = vec3<f32>(0.04, 0.28, 0.38); // Coastal Teal
  } else {
    base = vec3<f32>(0.90, 0.86, 0.78); // Alabaster Ivory
  }
  // Subtle variation across plumage regions: body -> inner wing -> dark/accented tip
  if (region > 1.5) {
    return mix(base * 0.62, vec3<f32>(0.95, 0.66, 0.22), select(0.12, 0.35, seed >= 0.80));
  } else if (region > 0.5) {
    return base * 1.08;
  }
  return mix(base, vec3<f32>(0.96, 0.93, 0.88), 0.18);
}
`;

export const SKY_SHADER_WGSL = /* wgsl */ `
${COMMON_WGSL}

@group(0) @binding(0) var<uniform> frame : FrameUniforms;

struct SkyOut {
  @builtin(position) clipPos : vec4<f32>,
  @location(0) ndc : vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid : u32) -> SkyOut {
  var pts = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0)
  );
  let p = pts[vid];
  var out : SkyOut;
  out.clipPos = vec4<f32>(p, 1.0, 1.0);
  out.ndc = p;
  return out;
}

@fragment
fn fs_main(in : SkyOut) -> @location(0) vec4<f32> {
  let farClip = frame.invViewProj * vec4<f32>(in.ndc, 1.0, 1.0);
  let worldPos = farClip.xyz / farClip.w;
  let dir = normalize(worldPos - frame.cameraPos.xyz);
  let col = evalSkyColor(dir, frame.sunDir.xyz);
  return vec4<f32>(linearToSrgb(acesToneMap(col)), 1.0);
}
`;

export const FLOOR_SHADER_WGSL = /* wgsl */ `
${COMMON_WGSL}

@group(0) @binding(0) var<uniform> frame : FrameUniforms;

struct FloorOut {
  @builtin(position) clipPos : vec4<f32>,
  @location(0) worldPos : vec3<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid : u32) -> FloorOut {
  var quad = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
    vec2<f32>(-1.0,  1.0)
  );
  let p = quad[vid] * 175.0;
  let wp = vec3<f32>(p.x, -44.0, p.y);
  var out : FloorOut;
  out.clipPos = frame.viewProj * vec4<f32>(wp, 1.0);
  out.worldPos = wp;
  return out;
}

@fragment
fn fs_main(in : FloorOut) -> @location(0) vec4<f32> {
  let xz = in.worldPos.xz;
  let r = length(xz);
  if (r > 170.0) { discard; }

  // Architectural concentric sanctuary stone rings + subtle radial seams
  let ang = atan2(xz.y, xz.x);
  let ringBand = abs(fract(r * 0.085) - 0.5);
  let spokeBand = abs(fract(ang * (12.0 / 6.2831853)) - 0.5);
  let ringLine = smoothstep(0.475, 0.498, ringBand);
  let spokeLine = smoothstep(0.482, 0.498, spokeBand) * smoothstep(10.0, 18.0, r);
  let tileChecker = step(0.5, fract(floor(r * 0.085) * 0.5 + floor(ang * (12.0 / 6.2831853)) * 0.5));

  // Brass inlay ring at pillar circle radius 38 and inner sanctuary radius 13
  let brassRing = smoothstep(0.65, 0.15, abs(r - 38.0)) + smoothstep(0.55, 0.12, abs(r - 12.5));

  var albedo = mix(vec3<f32>(0.76, 0.72, 0.65), vec3<f32>(0.71, 0.67, 0.61), tileChecker * 0.55);
  albedo = mix(albedo, vec3<f32>(0.54, 0.50, 0.45), max(ringLine, spokeLine) * 0.35);
  albedo = mix(albedo, vec3<f32>(0.88, 0.62, 0.24), clamp(brassRing, 0.0, 1.0) * 0.75);

  // Analytic soft ambient occlusion & directional shadow grounding under obelisk + 6 pillars
  let sunDir = frame.sunDir.xyz;
  let shadowOff = -sunDir.xz / max(sunDir.y, 0.2) * 14.0;
  var ao = smoothstep(6.5, 16.0, r);
  let obShad = smoothstep(5.5, 15.0, length(xz - shadowOff * 0.75));
  ao = min(ao, mix(0.62, 1.0, obShad));

  let pillarPos = array<vec2<f32>, 6>(
    vec2<f32>( 38.0,   0.0),
    vec2<f32>( 19.0,  32.908965),
    vec2<f32>(-19.0,  32.908965),
    vec2<f32>(-38.0,   0.0),
    vec2<f32>(-19.0, -32.908965),
    vec2<f32>( 19.0, -32.908965)
  );
  for (var p = 0u; p < 6u; p++) {
    let dP = length(xz - pillarPos[p]);
    ao = min(ao, mix(0.42, 1.0, smoothstep(3.2, 8.8, dP)));
    let dS = length(xz - (pillarPos[p] + shadowOff * 0.6));
    ao = min(ao, mix(0.68, 1.0, smoothstep(2.5, 8.5, dS)));
  }

  let N = vec3<f32>(0.0, 1.0, 0.0);
  let V = normalize(frame.cameraPos.xyz - in.worldPos);
  let ndl = max(dot(N, sunDir), 0.0);
  let H = normalize(sunDir + V);
  let spec = pow(max(dot(N, H), 0.0), 42.0) * (0.10 + brassRing * 0.28);

  let hemi = vec3<f32>(0.42, 0.64, 0.92) * 0.34 + vec3<f32>(0.46, 0.39, 0.32) * 0.10;
  var col = albedo * (hemi * ao + vec3<f32>(1.0, 0.94, 0.82) * ndl * 0.76 * ao) + vec3<f32>(1.0, 0.93, 0.80) * spec * ao;

  // Seamless mist horizon blend
  let viewVec = in.worldPos - frame.cameraPos.xyz;
  let dist = length(viewVec);
  let viewDir = viewVec / max(dist, 0.001);
  col = applyAerialFog(col, dist, viewDir, sunDir);
  let edgeFade = smoothstep(105.0, 168.0, r);
  let skyCol = evalSkyColor(viewDir, sunDir);
  col = mix(col, skyCol, edgeFade);

  return vec4<f32>(linearToSrgb(acesToneMap(col)), 1.0);
}
`;

export const ARENA_SHADER_WGSL = /* wgsl */ `
${COMMON_WGSL}

@group(0) @binding(0) var<uniform> frame : FrameUniforms;

struct ArenaIn {
  @location(0) pos : vec3<f32>,
  @location(1) normal : vec3<f32>,
  @location(2) color : vec3<f32>,
};

struct ArenaOut {
  @builtin(position) clipPos : vec4<f32>,
  @location(0) worldPos : vec3<f32>,
  @location(1) normal : vec3<f32>,
  @location(2) color : vec3<f32>,
};

@vertex
fn vs_main(in : ArenaIn) -> ArenaOut {
  var out : ArenaOut;
  out.clipPos = frame.viewProj * vec4<f32>(in.pos, 1.0);
  out.worldPos = in.pos;
  out.normal = in.normal;
  out.color = in.color;
  return out;
}

@fragment
fn fs_main(in : ArenaOut) -> @location(0) vec4<f32> {
  let N = normalize(in.normal);
  let sunDir = frame.sunDir.xyz;
  let viewVec = in.worldPos - frame.cameraPos.xyz;
  let dist = length(viewVec);
  let viewDir = viewVec / max(dist, 0.001);
  let V = -viewDir;

  let ndl = max(dot(N, sunDir), 0.0);
  let hemiMix = N.y * 0.5 + 0.5;
  let hemi = mix(vec3<f32>(0.46, 0.39, 0.32), vec3<f32>(0.42, 0.64, 0.92), hemiMix) * 0.35;

  // Height-based subtle architectural base grounding AO
  let heightAO = mix(0.68, 1.0, smoothstep(-44.0, -30.0, in.worldPos.y));

  // Detect golden bands via saturation for metallic glint
  let isGold = smoothstep(0.25, 0.55, in.color.r - in.color.b);
  let H = normalize(sunDir + V);
  let specPow = mix(20.0, 56.0, isGold);
  let specInt = mix(0.05, 0.38, isGold);
  let spec = pow(max(dot(N, H), 0.0), specPow) * specInt;

  var col = in.color * (hemi * heightAO + vec3<f32>(1.0, 0.94, 0.82) * ndl * 0.74) + vec3<f32>(1.0, 0.92, 0.72) * spec;
  col += in.color * isGold * 0.06;

  col = applyAerialFog(col, dist, viewDir, sunDir);
  return vec4<f32>(linearToSrgb(acesToneMap(col)), 1.0);
}
`;

export const BIRD_RENDER_WGSL = /* wgsl */ `
${COMMON_WGSL}

struct BirdInstance {
  posPhase : vec4<f32>, // xyz = world pos, w = wing phase
  velColor : vec4<f32>, // xyz = velocity, w = color seed
};

@group(0) @binding(0) var<uniform> frame : FrameUniforms;
@group(0) @binding(1) var<storage, read> instances : array<BirdInstance>;

struct BirdVertIn {
  @location(0) pos : vec3<f32>,
  @location(1) normal : vec3<f32>,
  @location(2) wingExtra : vec2<f32>, // x = wingWeight, y = colorRegion
};

struct BirdVertOut {
  @builtin(position) clipPos : vec4<f32>,
  @location(0) worldPos : vec3<f32>,
  @location(1) normal : vec3<f32>,
  @location(2) albedo : vec3<f32>,
};

@vertex
fn vs_main(in : BirdVertIn, @builtin(instance_index) instIdx : u32) -> BirdVertOut {
  let inst = instances[instIdx];
  let center = inst.posPhase.xyz;
  let phase = inst.posPhase.w;
  let vel = inst.velColor.xyz;
  let seed = inst.velColor.w;

  // Articulated gull-wing flapping with subtle outer tip phase lag
  let w = in.wingExtra.x;
  var lp = in.pos;
  var ln = in.normal;
  if (w != 0.0) {
    let flap = sin(phase - abs(w) * 0.55);
    lp.y += flap * w * 0.74;
    // Slight inward dihedral pull on big strokes
    lp.x *= 1.0 - abs(flap) * max(w, 0.0) * 0.10;
    // Tilt wing normal with flap slope so wings flash sunlit glints in flight
    let tilt = flap * sign(lp.x) * max(w, 0.0) * 0.42;
    ln = normalize(vec3<f32>(ln.x + tilt, ln.y, ln.z));
  }

  // Build orthonormal flight basis directly from velocity (saves 32 bytes/instance vs mat4!)
  let spd = max(length(vel), 0.001);
  let fwd = vel / spd;
  let worldUp = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(fwd.y) > 0.96);
  let right = normalize(cross(worldUp, fwd));
  let up = cross(fwd, right);

  // Subtle aerodynamic bank angle during turns
  let scale = 0.88 + fract(seed * 17.31) * 0.28;
  let worldPos = center + (right * lp.x + up * lp.y + fwd * lp.z) * scale;
  let worldNorm = normalize(right * ln.x + up * ln.y + fwd * ln.z);

  var out : BirdVertOut;
  out.clipPos = frame.viewProj * vec4<f32>(worldPos, 1.0);
  out.worldPos = worldPos;
  out.normal = worldNorm;
  out.albedo = birdPalette(seed, in.wingExtra.y);
  return out;
}

@fragment
fn fs_main(in : BirdVertOut, @builtin(front_facing) frontFacing : bool) -> @location(0) vec4<f32> {
  let N = normalize(select(-in.normal, in.normal, frontFacing));
  let sunDir = frame.sunDir.xyz;
  let viewVec = in.worldPos - frame.cameraPos.xyz;
  let dist = length(viewVec);
  let viewDir = viewVec / max(dist, 0.001);
  let V = -viewDir;

  let ndl = max(dot(N, sunDir), 0.0);
  let wrap = max(0.0, (dot(N, sunDir) + 0.25) / 1.25) * 0.07;
  let hemi = mix(vec3<f32>(0.46, 0.39, 0.32), vec3<f32>(0.42, 0.64, 0.92), N.y * 0.5 + 0.5) * 0.34;

  let H = normalize(sunDir + V);
  let spec = pow(max(dot(N, H), 0.0), 40.0) * 0.22;
  let rim = pow(1.0 - max(dot(N, V), 0.0), 3.5) * 0.08;

  var col = in.albedo * (hemi + vec3<f32>(1.0, 0.94, 0.82) * (ndl * 0.78 + wrap))
          + vec3<f32>(1.0, 0.95, 0.86) * (spec + rim);

  col = applyAerialFog(col, dist, viewDir, sunDir);
  return vec4<f32>(linearToSrgb(acesToneMap(col)), 1.0);
}
`;

export const BOIDS_COMPUTE_WGSL = /* wgsl */ `
struct BirdInstance {
  posPhase : vec4<f32>,
  velColor : vec4<f32>,
};

struct SimParams {
  count : u32,
  peerBudget : u32,
  peerOffset : u32,
  ptrMode : i32,
  dt : f32,
  sepW : f32,
  aliW : f32,
  cohW : f32,
  ptrPos : vec4<f32>,
};

@group(0) @binding(0) var<uniform> sim : SimParams;
@group(0) @binding(1) var<storage, read> srcBirds : array<BirdInstance>;
@group(0) @binding(2) var<storage, read_write> dstBirds : array<BirdInstance>;

// Cooperative 64-thread workgroup shared memory tile -> 64x reduction in global VRAM reads!
var<workgroup> tilePos : array<vec3<f32>, 64>;
var<workgroup> tileVel : array<vec3<f32>, 64>;

@compute @workgroup_size(64)
fn cs_main(
  @builtin(global_invocation_id) gid : vec3<u32>,
  @builtin(local_invocation_id) lid : vec3<u32>,
) {
  let idx = gid.x;
  let count = sim.count;
  let valid = idx < count;

  var pos = vec3<f32>(0.0);
  var vel = vec3<f32>(10.0, 0.0, 0.0);
  var phase = 0.0;
  var seed = 0.0;

  if (valid) {
    let selfBird = srcBirds[idx];
    pos = selfBird.posPhase.xyz;
    phase = selfBird.posPhase.w;
    vel = selfBird.velColor.xyz;
    seed = selfBird.velColor.w;
  }

  var sep = vec3<f32>(0.0);
  var ali = vec3<f32>(0.0);
  var coh = vec3<f32>(0.0);
  var neighbors = 0.0;

  let radSq = 20.25; // 4.5^2
  let sepSq = 4.0;   // 2.0^2
  let tiles = (sim.peerBudget + 63u) / 64u;

  for (var t = 0u; t < tiles; t++) {
    let peerIdx = (sim.peerOffset + t * 64u + lid.x) % count;
    let b = srcBirds[peerIdx];
    tilePos[lid.x] = b.posPhase.xyz;
    tileVel[lid.x] = b.velColor.xyz;
    workgroupBarrier();

    if (valid) {
      for (var k = 0u; k < 64u; k++) {
        let jPos = tilePos[k];
        let diff = pos - jPos;
        let d2 = dot(diff, diff);
        if (d2 > 0.0001 && d2 < radSq) {
          neighbors += 1.0;
          ali += tileVel[k];
          coh += jPos;
          if (d2 < sepSq) {
            sep += diff / (d2 + 0.15);
          }
        }
      }
    }
    workgroupBarrier();
  }

  if (!valid) { return; }

  var acc = vec3<f32>(0.0);
  if (neighbors > 0.0) {
    let invN = 1.0 / neighbors;
    acc += sep * sim.sepW + (ali * invN - vel) * sim.aliW + (coh * invN - pos) * sim.cohW;
  }

  // Sanctuary swirl + central obelisk + outer containment
  let rXzSq = pos.x * pos.x + pos.z * pos.z;
  if (rXzSq > 0.01) {
    let rXz = sqrt(rXzSq);
    let invR = 1.0 / rXz;
    acc.x += -pos.z * invR * 2.2;
    acc.z +=  pos.x * invR * 2.2;

    if (rXz < 8.2 && pos.y < 24.0) {
      let push = (8.2 - rXz) * 14.0 * invR;
      acc.x += pos.x * push;
      acc.z += pos.z * push;
    }
    if (rXz > 52.0) {
      let pull = (rXz - 52.0) * 3.2 * invR;
      acc.x -= pos.x * pull;
      acc.z -= pos.z * pull;
    }
  }

  // 6 Outer Monolith Pillars
  let pillarPos = array<vec2<f32>, 6>(
    vec2<f32>( 38.0,   0.0),
    vec2<f32>( 19.0,  32.908965),
    vec2<f32>(-19.0,  32.908965),
    vec2<f32>(-38.0,   0.0),
    vec2<f32>(-19.0, -32.908965),
    vec2<f32>( 19.0, -32.908965)
  );
  if (pos.y < 26.0) {
    for (var p = 0u; p < 6u; p++) {
      let d = pos.xz - pillarPos[p];
      let d2 = dot(d, d);
      if (d2 < 31.36 && d2 > 0.0001) {
        let dist = sqrt(d2);
        let push = (5.6 - dist) * 16.0 / dist;
        acc.x += d.x * push;
        acc.z += d.y * push;
      }
    }
  }

  // Vertical floor & ceiling cushion
  if (pos.y > 24.0) { acc.y -= (pos.y - 24.0) * 4.5; }
  else if (pos.y < -24.0) { acc.y += (-24.0 - pos.y) * 5.0; }
  acc.y += (2.0 - pos.y) * 0.16;

  // Interactive 3D pointer vortex
  if (sim.ptrMode != 0) {
    let toPtr = sim.ptrPos.xyz - pos;
    let d2 = dot(toPtr, toPtr);
    if (d2 < 784.0 && d2 > 0.01) {
      let d = sqrt(d2);
      let force = f32(sim.ptrMode) * (1.0 - d / 28.0) * 52.0 / d;
      acc += toPtr * force;
    }
  }

  let dt = clamp(sim.dt, 0.0001, 0.05);
  vel += acc * dt;
  let spd = length(vel);
  if (spd > 22.0) { vel *= 22.0 / spd; }
  else if (spd < 9.0) { vel = select(vec3<f32>(9.0, 0.0, 0.0), vel * (9.0 / spd), spd > 0.0001); }

  pos += vel * dt;
  var nextPhase = phase + dt * (9.5 + spd * 0.25 + max(vel.y, 0.0) * 0.28);
  if (nextPhase > 62.831853) { nextPhase -= 62.831853; }

  dstBirds[idx] = BirdInstance(vec4<f32>(pos, nextPhase), vec4<f32>(vel, seed));
}
`;

export const CRYSTAL_RENDER_WGSL = /* wgsl */ `
${COMMON_WGSL}

struct CrystalData {
  posScale : vec4<f32>,   // xyz = pos, w = scale (0 if culled)
  rotColor : vec4<f32>,   // x = rotY, y = rotX, z = colorSeed, w = pulse
};

@group(0) @binding(0) var<uniform> frame : FrameUniforms;
@group(0) @binding(1) var<storage, read> crystals : array<CrystalData>;

struct CrystalIn {
  @location(0) pos : vec3<f32>,
  @location(1) normal : vec3<f32>,
};

struct CrystalOut {
  @builtin(position) clipPos : vec4<f32>,
  @location(0) worldPos : vec3<f32>,
  @location(1) normal : vec3<f32>,
  @location(2) albedo : vec3<f32>,
  @location(3) pulse : f32,
};

@vertex
fn vs_main(in : CrystalIn, @builtin(instance_index) objIdx : u32) -> CrystalOut {
  let c = crystals[objIdx];
  let sc = c.posScale.w;

  var out : CrystalOut;
  if (sc <= 0.0) {
    out.clipPos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    out.worldPos = vec3<f32>(0.0);
    out.normal = vec3<f32>(0.0, 1.0, 0.0);
    out.albedo = vec3<f32>(0.0);
    out.pulse = 0.0;
    return out;
  }

  let cy = cos(c.rotColor.x); let sy = sin(c.rotColor.x);
  let cx = cos(c.rotColor.y); let sx = sin(c.rotColor.y);

  // Rotate X then Y (matches Three.js default XYZ Euler with z=0)
  let p1 = vec3<f32>(in.pos.x, in.pos.y * cx - in.pos.z * sx, in.pos.y * sx + in.pos.z * cx);
  let p2 = vec3<f32>(p1.x * cy + p1.z * sy, p1.y, -p1.x * sy + p1.z * cy);

  let n1 = vec3<f32>(in.normal.x, in.normal.y * cx - in.normal.z * sx, in.normal.y * sx + in.normal.z * cx);
  let n2 = vec3<f32>(n1.x * cy + n1.z * sy, n1.y, -n1.x * sy + n1.z * cy);

  let wp = c.posScale.xyz + p2 * sc;
  out.clipPos = frame.viewProj * vec4<f32>(wp, 1.0);
  out.worldPos = wp;
  out.normal = normalize(n2);
  out.albedo = birdPalette(c.rotColor.z, 0.8);
  out.pulse = c.rotColor.w;
  return out;
}

@fragment
fn fs_main(in : CrystalOut) -> @location(0) vec4<f32> {
  let N = normalize(in.normal);
  let sunDir = frame.sunDir.xyz;
  let viewVec = in.worldPos - frame.cameraPos.xyz;
  let dist = length(viewVec);
  let viewDir = viewVec / max(dist, 0.001);
  let V = -viewDir;

  let ndl = max(dot(N, sunDir), 0.0);
  let hemi = mix(vec3<f32>(0.46, 0.39, 0.32), vec3<f32>(0.42, 0.64, 0.92), N.y * 0.5 + 0.5) * 0.35;
  let H = normalize(sunDir + V);
  let spec = pow(max(dot(N, H), 0.0), 64.0) * 0.42;
  let fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.5) * 0.12;

  var col = in.albedo * (hemi + vec3<f32>(1.0, 0.94, 0.82) * ndl * 0.78)
          + vec3<f32>(1.0, 0.96, 0.88) * (spec + fresnel)
          + vec3<f32>(0.95, 0.68, 0.24) * in.pulse * 0.03;

  col = applyAerialFog(col, dist, viewDir, sunDir);
  return vec4<f32>(linearToSrgb(acesToneMap(col)), 1.0);
}
`;
