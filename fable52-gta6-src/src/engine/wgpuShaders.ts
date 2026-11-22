// Hand-crafted WGSL shaders for the custom Osaka City WebGPU engine.
// Features:
//  - GPU Compute frustum culling + drawIndexedIndirect / drawIndirect for InstancedMesh
//  - Unified PBR (static / instanced / skinned) with hardware texture_depth_2d_array CSM
//  - Zero-resolve multisampled Compute SSAO (symmetric paired-sample crease AO, full f32 depth)
//  - Compute Bloom Extract + Separable Gaussian Blur + Stephen Hill AP1 ACES Composite
//  - Instanced screen-aligned billboard quad Points & native LineSegments
//  - Gamma-aware GPU mipmap chain generator

import {
  DrawMode,
  FLAG_ALPHA_TO_COV,
  FLAG_AO_MAP,
  FLAG_BUMP_AO_UV1,
  FLAG_BUMP_MAP,
  FLAG_CAR_SHELL,
  FLAG_CRM_MAP,
  FLAG_EMISSIVE_MAP,
  FLAG_FLAT_SHADING,
  FLAG_FOG,
  FLAG_PERSON_OUTFIT,
  FLAG_RECEIVE_SHADOW,
  FLAG_TONEMAP_INLINE,
  FLAG_UNLIT,
  FLAG_VERTEX_COLORS,
  MAX_CASCADES,
} from './shaders';

const WGSL_ACES = /* wgsl */ `
fn rrtAndOdtFit(v: vec3f) -> vec3f {
  let a = v * (v + vec3f(0.0245786)) - vec3f(0.000090537);
  let b = v * (0.983729 * v + vec3f(0.4329510)) + vec3f(0.238081);
  return a / b;
}
fn acesToneMap(colIn: vec3f) -> vec3f {
  let ACESInputMat = mat3x3f(
    vec3f(0.59719, 0.07600, 0.02840),
    vec3f(0.35458, 0.90834, 0.13383),
    vec3f(0.04823, 0.01566, 0.83777)
  );
  let ACESOutputMat = mat3x3f(
    vec3f( 1.60475, -0.10208, -0.00327),
    vec3f(-0.53108,  1.10813, -0.07276),
    vec3f(-0.07367, -0.00605,  1.07602)
  );
  var c = colIn / 0.6;
  c = ACESInputMat * c;
  c = rrtAndOdtFit(c);
  c = ACESOutputMat * c;
  return clamp(c, vec3f(0.0), vec3f(1.0));
}
fn linearToSrgb(c: vec3f) -> vec3f {
  let lo = c * 12.92;
  let hi = 1.055 * pow(max(c, vec3f(0.0)), vec3f(1.0 / 2.4)) - vec3f(0.055);
  return select(lo, hi, c >= vec3f(0.0031308));
}
`;

const WGSL_SCENE_STRUCT = /* wgsl */ `
struct SceneUniforms {
  viewProj: mat4x4f,
  view: mat4x4f,
  camPos: vec4f,             // xyz = camPos, w = exposure
  sunDir: vec4f,             // xyz = sunDir, w = envBase
  sunColor: vec4f,           // xyz = sunColor * int, w = cascadeCount
  hemiSky: vec4f,            // xyz = hemiSky, w = fogFar
  hemiGround: vec4f,         // xyz = hemiGround, w = pixelScale
  skyTop: vec4f,             // xyz = skyTop, w = viewportW
  skyHorizon: vec4f,         // xyz = skyHorizon, w = viewportH
  fogColorNear: vec4f,       // xyz = fogColor, w = fogNear
  spotPosRange: vec4f,       // xyz = pos, w = range
  spotDirCos: vec4f,         // xyz = dir, w = cosAngle
  spotColorPen: vec4f,       // xyz = color * int, w = penCos
  pointPosRange: vec4f,      // xyz = pos, w = range
  pointColorDecay: vec4f,    // xyz = color * int, w = pointDecay
  spotMisc: vec4f,           // x = spotDecay, y = shadowMapSize, z = 0, w = 0
  cascadeSplits: vec4f,
  shadowBiases: vec4f,
  shadowDepthBiases: vec4f,
  shadowMats: array<mat4x4f, ${MAX_CASCADES}>,
};

struct DrawData {
  model: mat4x4f,
  baseColor: vec4f,          // rgb, opacity
  emissive: vec4f,           // rgb * int, alphaTest
  pbrParams: vec4f,          // rough, metal, clearcoat, ccRough
  detailParams: vec4f,       // bumpScale, aoInt, envInt, unused
  uvTransform: vec4f,        // repeat.xy, offset.xy
  personTop: vec4f,
  personBottom: vec4f,
  personRef: vec2f,
  flags: u32,
  boneOffset: u32,
};
`;

// =========================================================================
// GPU Compute Frustum Culling + Indirect Draw Compaction
// =========================================================================

export const WGSL_CULL_COMPUTE = /* wgsl */ `
struct CullParams {
  planes: array<vec4f, 6>,
  totalCount: u32,
  slotIndex: u32,
  visOffset: u32,
  _pad: u32,
};

@group(0) @binding(0) var<uniform> uCull: CullParams;
@group(0) @binding(1) var<storage, read> uSpheres: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> uVisIndices: array<u32>;
@group(0) @binding(3) var<storage, read_write> uIndirect: array<atomic<u32>>;

@compute @workgroup_size(64)
fn csMain(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= uCull.totalCount) { return; }
  let s = uSpheres[idx];
  let center = s.xyz;
  let r = s.w;
  for (var i = 0u; i < 6u; i = i + 1u) {
    let p = uCull.planes[i];
    if (dot(p.xyz, center) + p.w < -r) {
      return;
    }
  }
  let outSlot = atomicAdd(&uIndirect[uCull.slotIndex * 5u + 1u], 1u);
  uVisIndices[uCull.visOffset + outSlot] = idx;
}
`;

// =========================================================================
// Unified PBR Shader (Static / Instanced / Skinned)
// =========================================================================

export function buildWgslPbr(mode: DrawMode): string {
  const isInst = mode === 'instanced';
  const isSkin = mode === 'skinned';
  return /* wgsl */ `
${WGSL_SCENE_STRUCT}
${WGSL_ACES}

@group(0) @binding(0) var<uniform> uScene: SceneUniforms;
@group(0) @binding(1) var uShadowMap: texture_depth_2d_array;
@group(0) @binding(2) var uShadowSampler: sampler_comparison;

@group(1) @binding(0) var<storage, read> uDraws: array<DrawData>;
@group(1) @binding(1) var<storage, read> uBones: array<mat4x4f>;

@group(2) @binding(0) var uMap: texture_2d<f32>;
@group(2) @binding(1) var uMapSampler: sampler;
@group(2) @binding(2) var uEmissiveMap: texture_2d<f32>;
@group(2) @binding(3) var uCrmMap: texture_2d<f32>;
@group(2) @binding(4) var uBumpAoMap: texture_2d<f32>;
@group(2) @binding(5) var uPersonMask: texture_2d<f32>;
@group(2) @binding(6) var uPersonOver: texture_2d<f32>;
@group(2) @binding(7) var uRepeatSampler: sampler;
@group(2) @binding(8) var uClampSampler: sampler;
${isInst ? `
struct InstSlotUniform {
  drawIdx: u32,
  visOffset: u32,
  hasInstCol: u32,
  _pad: u32,
};
@group(3) @binding(0) var<storage, read> uInstMats: array<mat4x4f>;
@group(3) @binding(1) var<storage, read> uInstCols: array<vec4f>;
@group(3) @binding(2) var<storage, read> uVisIndices: array<u32>;
@group(3) @binding(3) var<uniform> uInstSlot: InstSlotUniform;
` : ''}

struct VertIn {
  @location(0) aPosition: vec3f,
  @location(1) aNormal: vec3f,
  @location(2) aUv: vec2f,
  @location(3) aUv1: vec2f,
  @location(4) aColor: vec4f,
${isSkin ? `
  @location(5) aSkinIndex: vec4f,
  @location(6) aSkinWeight: vec4f,
` : ''}
};

struct VertOut {
  @builtin(position) pos: vec4f,
  @location(0) worldPos: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
  @location(3) uv1: vec2f,
  @location(4) color: vec4f,
  @location(5) viewDepth: f32,
  @location(6) @interpolate(flat) drawIdx: u32,
};

@vertex
fn vsMain(inp: VertIn, @builtin(instance_index) instIdx: u32) -> VertOut {
${isInst ? `
  let drawIdx = uInstSlot.drawIdx;
  let draw = uDraws[drawIdx];
  let actualInst = uVisIndices[uInstSlot.visOffset + instIdx];
  let instMat = uInstMats[actualInst];
  let worldMat = draw.model * instMat;
  let instCol = select(vec4f(1.0), uInstCols[actualInst], uInstSlot.hasInstCol != 0u);
` : `
  let drawIdx = instIdx;
  let draw = uDraws[drawIdx];
  let worldMat = draw.model;
`}
  var localPos = vec4f(inp.aPosition, 1.0);
  var localNor = inp.aNormal;
${isSkin ? `
  let bo = draw.boneOffset;
  let b0 = uBones[bo + u32(inp.aSkinIndex.x)];
  let b1 = uBones[bo + u32(inp.aSkinIndex.y)];
  let b2 = uBones[bo + u32(inp.aSkinIndex.z)];
  let b3 = uBones[bo + u32(inp.aSkinIndex.w)];
  let skinMat = b0 * inp.aSkinWeight.x + b1 * inp.aSkinWeight.y + b2 * inp.aSkinWeight.z + b3 * inp.aSkinWeight.w;
  localPos = skinMat * localPos;
  localNor = (mat3x3f(skinMat[0].xyz, skinMat[1].xyz, skinMat[2].xyz)) * localNor;
` : ''}
  let wp = worldMat * localPos;
  let nm = mat3x3f(worldMat[0].xyz, worldMat[1].xyz, worldMat[2].xyz);
  let sx = vec3f(dot(nm[0], nm[0]), dot(nm[1], nm[1]), dot(nm[2], nm[2]));
  let invS2 = vec3f(1.0) / max(sx, vec3f(1e-8));
  let norMat = mat3x3f(nm[0] * invS2.x, nm[1] * invS2.y, nm[2] * invS2.z);
  let wn = norMat * localNor;
  let wnl = length(wn);

  var out: VertOut;
  out.worldPos = wp.xyz;
  out.normal = select(vec3f(0.0, 1.0, 0.0), wn / wnl, wnl > 1e-6);
  out.uv = inp.aUv * draw.uvTransform.xy + draw.uvTransform.zw;
  out.uv1 = inp.aUv1;
  out.color = inp.aColor${isInst ? ' * instCol' : ''};
  let vp = uScene.view * wp;
  out.viewDepth = -vp.z;
  out.pos = uScene.viewProj * wp;
  out.drawIdx = drawIdx;
  return out;
}

const PI: f32 = 3.14159265;

fn distGGX(NdotH: f32, rough: f32) -> f32 {
  let a = rough * rough;
  let a2 = a * a;
  let d = NdotH * NdotH * (a2 - 1.0) + 1.0;
  return a2 / max(PI * d * d, 1e-5);
}

fn geomSmith(NdotV: f32, NdotL: f32, rough: f32) -> f32 {
  let r = rough + 1.0;
  let k = (r * r) * 0.125;
  let gv = NdotV / (NdotV * (1.0 - k) + k);
  let gl = NdotL / (NdotL * (1.0 - k) + k);
  return gv * gl;
}

fn fresnelSchlick(cosT: f32, F0: vec3f) -> vec3f {
  let t = clamp(1.0 - cosT, 0.0, 1.0);
  let t2 = t * t;
  return F0 + (vec3f(1.0) - F0) * (t2 * t2 * t);
}

fn fresnelSchlickRoughness(cosT: f32, F0: vec3f, rough: f32) -> vec3f {
  let t = clamp(1.0 - cosT, 0.0, 1.0);
  let t2 = t * t;
  return F0 + (max(vec3f(1.0 - rough), F0) - F0) * (t2 * t2 * t);
}

fn physDistanceAtten(dist: f32, range: f32, decay: f32) -> f32 {
  var atten = 1.0 / max(pow(dist, max(decay, 0.01)), 0.01);
  if (range > 0.0) {
    let r = dist / range;
    let r2 = r * r;
    let win = clamp(1.0 - r2 * r2, 0.0, 1.0);
    atten *= win * win;
  }
  return atten;
}

fn evalDirectPBR(N: vec3f, V: vec3f, L: vec3f, NdotV: f32, NdotL: f32, albedo: vec3f, F0: vec3f, rough: f32, metal: f32, clearcoat: f32, ccRough: f32, radiance: vec3f) -> vec3f {
  let H = normalize(V + L);
  let NdotH = max(dot(N, H), 0.0);
  let HdotV = max(dot(H, V), 0.0);
  let F = fresnelSchlick(HdotV, F0);
  let D = distGGX(NdotH, rough);
  let G = geomSmith(NdotV, NdotL, rough);
  let spec = (D * G * F) / max(4.0 * NdotV * NdotL, 0.001);
  let kD = (vec3f(1.0) - F) * (1.0 - metal);
  var outCol = (kD * albedo / PI + spec) * radiance;
  if (clearcoat > 0.01) {
    let ccD = distGGX(NdotH, max(ccRough, 0.05));
    let ccF = 0.04 + 0.96 * pow(1.0 - HdotV, 5.0);
    outCol += vec3f(clearcoat * ccD * ccF * 0.25) * radiance;
  }
  return outCol;
}

fn sampleCascade(cIdx: i32, worldPos: vec3f, N: vec3f, NdotL: f32) -> f32 {
  let nBias = uScene.shadowBiases[cIdx] * (1.0 - 0.5 * clamp(NdotL, 0.0, 1.0));
  let sc = uScene.shadowMats[cIdx] * vec4f(worldPos + N * nBias, 1.0);
  let invW = 1.0 / sc.w;
  let uv = vec2f(sc.x * invW * 0.5 + 0.5, 0.5 - sc.y * invW * 0.5);
  let z = sc.z * invW;
  if (uv.x < 0.002 || uv.x > 0.998 || uv.y < 0.002 || uv.y > 0.998 || z < 0.0 || z > 0.999) {
    return 1.0;
  }
  let texel = vec2f(1.0 / max(uScene.spotMisc.y, 1.0));
  let slopeScale = 1.0 + 2.0 * (1.0 - clamp(NdotL, 0.0, 1.0));
  let refZ = clamp(z - uScene.shadowDepthBiases[cIdx] * slopeScale, 0.0, 1.0);
  var s = 0.0;
  s += textureSampleCompareLevel(uShadowMap, uShadowSampler, uv + vec2f(-0.7, -0.3) * texel, cIdx, refZ);
  s += textureSampleCompareLevel(uShadowMap, uShadowSampler, uv + vec2f( 0.7,  0.3) * texel, cIdx, refZ);
  s += textureSampleCompareLevel(uShadowMap, uShadowSampler, uv + vec2f(-0.3,  0.7) * texel, cIdx, refZ);
  s += textureSampleCompareLevel(uShadowMap, uShadowSampler, uv + vec2f( 0.3, -0.7) * texel, cIdx, refZ);
  return s * 0.25;
}

fn computeShadow(worldPos: vec3f, viewDepth: f32, N: vec3f, NdotL: f32) -> f32 {
  let cCount = i32(uScene.sunColor.w);
  if (cCount <= 0) { return 1.0; }
  var cIdx = cCount - 1;
  for (var i = 0; i < ${MAX_CASCADES}; i = i + 1) {
    if (i >= cCount) { break; }
    if (viewDepth < uScene.cascadeSplits[i]) {
      cIdx = i;
      break;
    }
  }
  let lastFar = uScene.cascadeSplits[cCount - 1];
  if (viewDepth > lastFar) { return 1.0; }
  var sh = sampleCascade(cIdx, worldPos, N, NdotL);
  if (cIdx < cCount - 1) {
    let splitFar = uScene.cascadeSplits[cIdx];
    let blendStart = splitFar * 0.88;
    if (viewDepth > blendStart) {
      let f = smoothstep(blendStart, splitFar, viewDepth);
      sh = mix(sh, sampleCascade(cIdx + 1, worldPos, N, NdotL), f);
    }
  } else {
    if (viewDepth > lastFar * 0.88) {
      sh = mix(sh, 1.0, smoothstep(lastFar * 0.88, lastFar, viewDepth));
    }
  }
  return sh;
}

fn evalAnalyticalSkyIBL(R: vec3f, rough: f32) -> vec3f {
  let y = clamp(R.y, -1.0, 1.0);
  let sky = mix(uScene.skyHorizon.xyz, uScene.skyTop.xyz, pow(clamp(y, 0.0, 1.0), 0.55));
  let ground = uScene.hemiGround.xyz * 0.45;
  var env = mix(ground, sky, smoothstep(-0.18 - rough * 0.35, 0.18 + rough * 0.35, y));
  let sDot = max(dot(R, uScene.sunDir.xyz), 0.0);
  let specPow = mix(96.0, 6.0, rough * rough);
  env += uScene.sunColor.xyz * pow(sDot, specPow) * mix(0.28, 0.03, rough);
  return env;
}

@fragment
fn fsMain(inp: VertOut, @builtin(front_facing) isFront: bool) -> @location(0) vec4f {
  let draw = uDraws[inp.drawIdx];
  let flags = draw.flags;

  // Evaluate ALL implicit-derivative textureSample & dpdx/dpdy/fwidth in strict uniform control flow
  let texCol = textureSample(uMap, uMapSampler, inp.uv);
  let emTex = textureSample(uEmissiveMap, uRepeatSampler, inp.uv).rgb;
  let crmTex = textureSample(uCrmMap, uRepeatSampler, inp.uv).rgb;
  let detailUv = select(inp.uv, inp.uv1, (flags & ${FLAG_BUMP_AO_UV1}u) != 0u);
  let bumpAoVal = textureSample(uBumpAoMap, uRepeatSampler, detailUv).r;
  let pm = textureSample(uPersonMask, uClampSampler, inp.uv);
  let po = textureSample(uPersonOver, uClampSampler, inp.uv);

  let dp1 = dpdx(inp.worldPos);
  let dp2 = dpdy(inp.worldPos);
  let dh1 = dpdx(bumpAoVal);
  let dh2 = dpdy(bumpAoVal);
  let fwTexA = fwidth(texCol.a);
  let rawDiffuse = vec4f(draw.baseColor.rgb * texCol.rgb, draw.baseColor.a * texCol.a);
  let vcDiffuse = rawDiffuse * select(vec4f(1.0), inp.color, (flags & ${FLAG_VERTEX_COLORS}u) != 0u);
  let fwDiffA = fwidth(vcDiffuse.a);

  var diffuse = rawDiffuse;
  var carLining = false;

  if ((flags & ${FLAG_CAR_SHELL}u) != 0u) {
    var cov = 1.0;
    if (draw.emissive.a > 0.0) {
      if ((flags & ${FLAG_ALPHA_TO_COV}u) != 0u) {
        cov = clamp((texCol.a - draw.emissive.a) / max(fwTexA, 1e-4) + 0.5, 0.0, 1.0);
        if (cov == 0.0) { discard; }
      } else if (texCol.a < draw.emissive.a) {
        discard;
      }
    }
    let paintMask = clamp((1.0 - texCol.a) * 2.5, 0.0, 1.0);
    carLining = inp.color.a < 0.5;
    diffuse = vec4f(
      select(texCol.rgb * mix(vec3f(1.0), inp.color.rgb, paintMask), inp.color.rgb, carLining),
      cov
    );
  } else {
    diffuse = vcDiffuse;
    if (draw.emissive.a > 0.0) {
      if ((flags & ${FLAG_ALPHA_TO_COV}u) != 0u) {
        diffuse.a = clamp((diffuse.a - draw.emissive.a) / max(fwDiffA, 1e-4) + 0.5, 0.0, 1.0);
        if (diffuse.a == 0.0) { discard; }
      } else if (diffuse.a < draw.emissive.a) {
        discard;
      }
    }
  }

  if ((flags & ${FLAG_PERSON_OUTFIT}u) != 0u) {
    let pl = dot(diffuse.rgb, vec3f(0.3333));
    diffuse = vec4f(
      mix(diffuse.rgb, draw.personTop.rgb * min(pl / max(draw.personRef.x, 1e-4), 2.2), pm.r * draw.personTop.a),
      diffuse.a
    );
    diffuse = vec4f(
      mix(diffuse.rgb, draw.personBottom.rgb * min(pl / max(draw.personRef.y, 1e-4), 2.2), pm.g * draw.personBottom.a),
      diffuse.a
    );
    diffuse = vec4f(
      mix(diffuse.rgb, po.rgb * (0.86 + 0.3 * pl), po.a),
      diffuse.a
    );
  }

  var color: vec3f;
  if ((flags & ${FLAG_UNLIT}u) != 0u) {
    color = diffuse.rgb;
  } else {
    // WebGPU framebuffer Y points downward, so cross(dp2, dp1) faces the camera!
    let flatN = cross(dp2, dp1);
    let flatLen = length(flatN);
    let smoothN = select(-normalize(inp.normal), normalize(inp.normal), isFront);
    var N = select(smoothN, select(smoothN, flatN / flatLen, flatLen > 1e-7), (flags & ${FLAG_FLAT_SHADING}u) != 0u);

    if ((flags & ${FLAG_BUMP_MAP}u) != 0u) {
      let r1 = cross(dp2, N);
      let r2 = cross(N, dp1);
      let det = dot(dp1, r1);
      if (abs(det) > 1e-6) {
        let surfGrad = sign(det) * (dh1 * r1 + dh2 * r2);
        N = normalize(abs(det) * N - draw.detailParams.x * 0.08 * surfGrad);
      }
    }

    var rough = draw.pbrParams.x;
    var metal = draw.pbrParams.y;
    var clearcoat = draw.pbrParams.z;
    let ccRough = draw.pbrParams.w;

    if ((flags & ${FLAG_CRM_MAP}u) != 0u) {
      clearcoat *= crmTex.r;
      rough *= crmTex.g;
      metal *= crmTex.b;
    }
    if (carLining) {
      rough = 0.92;
      metal = 0.0;
      clearcoat = 0.0;
    }
    rough = clamp(rough, 0.06, 1.0);
    metal = clamp(metal, 0.0, 1.0);

    let V = normalize(uScene.camPos.xyz - inp.worldPos);
    let NdotV = max(dot(N, V), 0.001);
    let F0 = mix(vec3f(0.04), diffuse.rgb, metal);

    // 1. Direct Sun / Moon light with hardware CSM
    let L = uScene.sunDir.xyz;
    let NdotL = max(dot(N, L), 0.0);
    var shadow = 1.0;
    if ((flags & ${FLAG_RECEIVE_SHADOW}u) != 0u && NdotL > 0.0) {
      shadow = computeShadow(inp.worldPos, inp.viewDepth, N, NdotL);
    }

    var Lo = vec3f(0.0);
    if (NdotL > 0.0) {
      Lo += evalDirectPBR(N, V, L, NdotV, NdotL, diffuse.rgb, F0, rough, metal, clearcoat, ccRough, uScene.sunColor.xyz * (shadow * NdotL));
    }

    // 2. Player car headlight SpotLight
    if (uScene.spotPosRange.w > 0.0 && (uScene.spotColorPen.r + uScene.spotColorPen.g + uScene.spotColorPen.b) > 0.001) {
      let toLight = uScene.spotPosRange.xyz - inp.worldPos;
      let dist = length(toLight);
      if (dist < uScene.spotPosRange.w) {
        let sL = toLight / max(dist, 0.001);
        let cosTheta = dot(-sL, uScene.spotDirCos.xyz);
        if (cosTheta > uScene.spotDirCos.w) {
          let spotCone = smoothstep(uScene.spotDirCos.w, uScene.spotColorPen.w, cosTheta);
          let atten = physDistanceAtten(dist, uScene.spotPosRange.w, uScene.spotMisc.x);
          let sNdotL = max(dot(N, sL), 0.0);
          if (sNdotL > 0.0) {
            Lo += evalDirectPBR(N, V, sL, NdotV, sNdotL, diffuse.rgb, F0, rough, metal, clearcoat, ccRough, uScene.spotColorPen.rgb * (spotCone * atten * sNdotL));
          }
        }
      }
    }

    // 3. Explosion blast PointLight
    if (uScene.pointPosRange.w > 0.0 && (uScene.pointColorDecay.r + uScene.pointColorDecay.g + uScene.pointColorDecay.b) > 0.001) {
      let toP = uScene.pointPosRange.xyz - inp.worldPos;
      let dist = length(toP);
      if (dist < uScene.pointPosRange.w) {
        let pL = toP / max(dist, 0.001);
        let atten = physDistanceAtten(dist, uScene.pointPosRange.w, uScene.pointColorDecay.w);
        let pNdotL = max(dot(N, pL), 0.0);
        if (pNdotL > 0.0) {
          Lo += evalDirectPBR(N, V, pL, NdotV, pNdotL, diffuse.rgb, F0, rough, metal, clearcoat, ccRough, uScene.pointColorDecay.rgb * (atten * pNdotL));
        }
      }
    }

    // 4. Hemisphere ambient + Analytical Sky IBL reflection
    var ao = 1.0;
    if ((flags & ${FLAG_AO_MAP}u) != 0u) {
      ao = mix(1.0, bumpAoVal, draw.detailParams.y);
    }
    let hemiCol = mix(uScene.hemiGround.xyz, uScene.hemiSky.xyz, N.y * 0.5 + 0.5);
    let Fenv = fresnelSchlickRoughness(NdotV, F0, rough);
    let kDenv = (vec3f(1.0) - Fenv) * (1.0 - metal);
    var ambient = kDenv * (diffuse.rgb / PI) * hemiCol * ao;

    let envScale = draw.detailParams.z * uScene.sunDir.w;
    if (envScale > 0.001) {
      let R = reflect(-V, N);
      var envSpec = evalAnalyticalSkyIBL(R, rough) * Fenv * mix(0.95, 0.12, rough) * envScale;
      if (clearcoat > 0.01) {
        let ccFres = 0.04 + 0.96 * pow(1.0 - NdotV, 5.0);
        envSpec += evalAnalyticalSkyIBL(R, ccRough) * (clearcoat * ccFres * envScale * 0.95);
      }
      ambient += envSpec * ao;
    }

    // 5. Emissive
    var em = draw.emissive.rgb;
    if ((flags & ${FLAG_EMISSIVE_MAP}u) != 0u) {
      em *= emTex;
    }

    color = Lo + ambient + em;
  }

  // Distance fog
  if ((flags & ${FLAG_FOG}u) != 0u) {
    let fogFactor = smoothstep(uScene.fogColorNear.w, uScene.hemiSky.w, inp.viewDepth);
    color = mix(color, uScene.fogColorNear.rgb, fogFactor);
  }

  if ((flags & ${FLAG_TONEMAP_INLINE}u) != 0u) {
    color = linearToSrgb(acesToneMap(color * uScene.camPos.w));
  }

  return vec4f(color, diffuse.a);
}
`;
}

// =========================================================================
// Depth-Only Shadow Shader (with Alpha Cutout Support)
// =========================================================================

export function buildWgslShadow(mode: DrawMode): string {
  const isInst = mode === 'instanced';
  const isSkin = mode === 'skinned';
  return /* wgsl */ `
${WGSL_SCENE_STRUCT}

struct ShadowPassUniforms {
  lightVP: mat4x4f,
};

@group(0) @binding(0) var<uniform> uShadow: ShadowPassUniforms;
@group(1) @binding(0) var<storage, read> uDraws: array<DrawData>;
@group(1) @binding(1) var<storage, read> uBones: array<mat4x4f>;
@group(2) @binding(0) var uMap: texture_2d<f32>;
@group(2) @binding(1) var uMapSampler: sampler;
${isInst ? `
struct InstSlotUniform {
  drawIdx: u32,
  visOffset: u32,
  hasInstCol: u32,
  _pad: u32,
};
@group(3) @binding(0) var<storage, read> uInstMats: array<mat4x4f>;
@group(3) @binding(1) var<storage, read> uInstCols: array<vec4f>;
@group(3) @binding(2) var<storage, read> uVisIndices: array<u32>;
@group(3) @binding(3) var<uniform> uInstSlot: InstSlotUniform;
` : ''}

struct VertIn {
  @location(0) aPosition: vec3f,
  @location(1) aNormal: vec3f,
  @location(2) aUv: vec2f,
  @location(3) aUv1: vec2f,
  @location(4) aColor: vec4f,
${isSkin ? `
  @location(5) aSkinIndex: vec4f,
  @location(6) aSkinWeight: vec4f,
` : ''}
};

struct VertOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) @interpolate(flat) alphaTest: f32,
};

@vertex
fn vsMain(inp: VertIn, @builtin(instance_index) instIdx: u32) -> VertOut {
${isInst ? `
  let drawIdx = uInstSlot.drawIdx;
  let draw = uDraws[drawIdx];
  let actualInst = uVisIndices[uInstSlot.visOffset + instIdx];
  let worldMat = draw.model * uInstMats[actualInst];
` : `
  let draw = uDraws[instIdx];
  let worldMat = draw.model;
`}
  var localPos = vec4f(inp.aPosition, 1.0);
${isSkin ? `
  let bo = draw.boneOffset;
  let b0 = uBones[bo + u32(inp.aSkinIndex.x)];
  let b1 = uBones[bo + u32(inp.aSkinIndex.y)];
  let b2 = uBones[bo + u32(inp.aSkinIndex.z)];
  let b3 = uBones[bo + u32(inp.aSkinIndex.w)];
  let skinMat = b0 * inp.aSkinWeight.x + b1 * inp.aSkinWeight.y + b2 * inp.aSkinWeight.z + b3 * inp.aSkinWeight.w;
  localPos = skinMat * localPos;
` : ''}
  var out: VertOut;
  out.pos = uShadow.lightVP * (worldMat * localPos);
  out.uv = inp.aUv * draw.uvTransform.xy + draw.uvTransform.zw;
  out.alphaTest = draw.emissive.a;
  return out;
}

@fragment
fn fsMain(inp: VertOut) {
  let a = textureSample(uMap, uMapSampler, inp.uv).a;
  if (inp.alphaTest > 0.0 && a < inp.alphaTest) {
    discard;
  }
}
`;
}

// =========================================================================
// Sky Dome Shader
// =========================================================================

export const WGSL_SKY = /* wgsl */ `
${WGSL_ACES}

struct SkyUniforms {
  viewProj: mat4x4f,
  model: mat4x4f,
  top: vec4f,        // rgb, stars
  horizon: vec4f,    // rgb, time
  sunDir: vec4f,     // xyz, cover
  sunColor: vec4f,   // rgb, moon
  cloudColor: vec4f, // rgb, exposure
  flags: vec4f,      // x = toneMapInline
};

@group(0) @binding(0) var<uniform> uSky: SkyUniforms;

struct VertIn {
  @location(0) aPosition: vec3f,
  @location(1) aNormal: vec3f,
  @location(2) aUv: vec2f,
  @location(3) aUv1: vec2f,
  @location(4) aColor: vec4f,
};

struct VertOut {
  @builtin(position) pos: vec4f,
  @location(0) dir: vec3f,
};

@vertex
fn vsMain(inp: VertIn) -> VertOut {
  var out: VertOut;
  out.dir = normalize(inp.aPosition);
  out.pos = uSky.viewProj * (uSky.model * vec4f(inp.aPosition, 1.0));
  return out;
}

fn hash3(p: vec3f) -> f32 { return fract(sin(dot(p, vec3f(12.9898, 78.233, 37.719))) * 43758.5453); }
fn hash2(p: vec2f) -> f32 { return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453); }
fn noise(p: vec2f) -> f32 {
  let i = floor(p);
  var f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(hash2(i), hash2(i + vec2f(1.0, 0.0)), f.x),
    mix(hash2(i + vec2f(0.0, 1.0)), hash2(i + vec2f(1.0, 1.0)), f.x),
    f.y
  );
}
fn fbm(pIn: vec2f) -> f32 {
  var p = pIn;
  var a = 0.5;
  var s = 0.0;
  for (var i = 0; i < 5; i = i + 1) {
    s += a * noise(p);
    p = p * 2.03 + vec2f(17.0, 9.0);
    a *= 0.5;
  }
  return s;
}

@fragment
fn fsMain(inp: VertOut) -> @location(0) vec4f {
  let d = normalize(inp.dir);
  let h = clamp(d.y, 0.0, 1.0);
  var col = mix(uSky.horizon.rgb, uSky.top.rgb, pow(h, 0.55));

  let stars = uSky.top.w;
  let time = uSky.horizon.w;
  let cover = uSky.sunDir.w;
  let moon = uSky.sunColor.w;
  let exposure = uSky.cloudColor.w;
  let sunDir = uSky.sunDir.xyz;

  var cloud = 0.0;
  var thick = 0.0;
  if (d.y > 0.0) {
    let uv = d.xz / (d.y + 0.14) * 1.15 + vec2f(time * 0.006, time * 0.0025);
    let n = fbm(uv);
    let edge = mix(0.66, 0.26, cover);
    cloud = smoothstep(edge, edge + 0.2, n) * smoothstep(0.0, 0.1, d.y);
    thick = smoothstep(edge + 0.1, edge + 0.45, n);
    let lit = mix(uSky.cloudColor.rgb, uSky.cloudColor.rgb * 0.55, thick * (0.4 + 0.5 * cover));
    col = mix(col, lit, cloud * 0.92);
  }

  let s = max(dot(d, sunDir), 0.0);
  let sunVis = (1.0 - cloud * 0.85) * (1.0 - cover * 0.85);
  col += uSky.sunColor.rgb * (pow(s, 600.0) * 3.0 + pow(s, 24.0) * 0.45) * sunVis;

  let moonDir = normalize(vec3f(-sunDir.x, max(0.35, -sunDir.y), -sunDir.z));
  let m = max(dot(d, moonDir), 0.0);
  let moonVis = stars * (1.0 - cloud * 0.9);
  let phase = 1.0 - abs(moon - 0.5) * 2.0;
  let disc = smoothstep(0.9994, 0.9997, m);
  if (disc > 0.0) {
    let right = normalize(cross(moonDir, vec3f(0.0, 1.0, 0.0)));
    let up = cross(right, moonDir);
    let p = vec2f(dot(d - moonDir, right), dot(d - moonDir, up)) / 0.032;
    let z = sqrt(max(0.0, 1.0 - dot(p, p)));
    let a = moon * 6.2831853;
    let toSun = vec3f(sin(a), 0.0, -cos(a));
    let litM = smoothstep(-0.08, 0.12, dot(vec3f(p, z), toSun));
    col += vec3f(1.5, 1.55, 1.7) * disc * (0.06 + 0.94 * litM) * moonVis;
  }
  col += vec3f(0.45, 0.55, 0.75) * pow(m, 40.0) * 0.35 * phase * moonVis;

  if (stars > 0.01 && d.y > 0.02) {
    let cell = floor(d * 260.0);
    let r = hash3(cell);
    if (r > 0.996) {
      let b = (r - 0.996) * 250.0 * stars * smoothstep(0.02, 0.25, d.y) * (1.0 - cloud);
      col += vec3f(0.85, 0.9, 1.0) * b;
    }
  }

  if (uSky.flags.x > 0.5) {
    col = linearToSrgb(acesToneMap(col * exposure));
  }
  return vec4f(col, 1.0);
}
`;

// =========================================================================
// Instanced Billboard Quad Points (WebGPU has no gl_PointSize > 1px!)
// =========================================================================

export const WGSL_POINTS = /* wgsl */ `
${WGSL_ACES}

struct PointsUniforms {
  viewProj: mat4x4f,
  view: mat4x4f,
  model: mat4x4f,
  params: vec4f, // x = pixelScale, y = invFbW, z = invFbH, w = exposure
  flags: vec4f,  // x = toneMapInline
};

@group(0) @binding(0) var<uniform> uPts: PointsUniforms;

struct PtInstIn {
  @location(0) aPosition: vec3f,
  @location(1) aSize: f32,
  @location(2) aColor: vec3f,
  @location(3) aAlpha: f32,
};

struct VertOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) color: vec3f,
  @location(2) alpha: f32,
};

@vertex
fn vsMain(inp: PtInstIn, @builtin(vertex_index) vid: u32) -> VertOut {
  var corners = array<vec2f, 6>(
    vec2f(-0.5, -0.5), vec2f( 0.5, -0.5), vec2f( 0.5,  0.5),
    vec2f(-0.5, -0.5), vec2f( 0.5,  0.5), vec2f(-0.5,  0.5)
  );
  let corner = corners[vid];
  let wp = uPts.model * vec4f(inp.aPosition, 1.0);
  let mv = uPts.view * wp;
  var clip = uPts.viewProj * wp;
  let ptPx = inp.aSize * 300.0 * uPts.params.x / max(1.0, -mv.z);
  let offsetNdc = corner * vec2f(ptPx * 2.0 * uPts.params.y, ptPx * 2.0 * uPts.params.z);
  clip = vec4f(clip.xy + offsetNdc * clip.w, clip.z, clip.w);

  var out: VertOut;
  out.pos = clip;
  out.uv = corner + vec2f(0.5);
  out.color = inp.aColor;
  out.alpha = inp.aAlpha;
  return out;
}

@fragment
fn fsMain(inp: VertOut) -> @location(0) vec4f {
  let d = length(inp.uv - vec2f(0.5));
  if (d > 0.5 || inp.alpha <= 0.001) { discard; }
  var c = inp.color;
  if (uPts.flags.x > 0.5) {
    c = linearToSrgb(acesToneMap(c * uPts.params.w));
  }
  return vec4f(c, smoothstep(0.5, 0.15, d) * inp.alpha);
}
`;

// =========================================================================
// LineSegments Shader (Bullet Tracers & Rain Streaks)
// =========================================================================

export const WGSL_LINES = /* wgsl */ `
${WGSL_ACES}

struct LineUniforms {
  viewProj: mat4x4f,
  model: mat4x4f,
  color: vec4f,
  flags: vec4f, // x = toneMapInline, y = exposure
};

@group(0) @binding(0) var<uniform> uLine: LineUniforms;

@vertex
fn vsMain(@location(0) aPosition: vec3f) -> @builtin(position) vec4f {
  return uLine.viewProj * (uLine.model * vec4f(aPosition, 1.0));
}

@fragment
fn fsMain() -> @location(0) vec4f {
  var c = uLine.color.rgb;
  if (uLine.flags.x > 0.5) {
    c = linearToSrgb(acesToneMap(c * uLine.flags.y));
  }
  return vec4f(c, uLine.color.a);
}
`;

// =========================================================================
// Compute Shader SSAO (Zero Depth-Resolve Pass — Direct MSAA or 1x Depth Read!)
// =========================================================================

export function buildWgslSsaoCompute(multisampled: boolean): string {
  return /* wgsl */ `
struct SsaoUniforms {
  fullTexel: vec2f,
  aoSize: vec2f,
  projParams: vec4f, // x = near, y = far, z = tanHalfFov * aspect, w = tanHalfFov
  samples: u32,
  _p0: u32,
  _p1: u32,
  _p2: u32,
};

@group(0) @binding(0) var<uniform> uAo: SsaoUniforms;
@group(0) @binding(1) var uDepth: ${multisampled ? 'texture_depth_multisampled_2d' : 'texture_depth_2d'};
@group(0) @binding(2) var uAoOut: texture_storage_2d<rgba8unorm, write>;

fn linDepthTexel(p: vec2i, maxP: vec2i) -> f32 {
  let d = textureLoad(uDepth, clamp(p, vec2i(0), maxP), 0);
  let near = uAo.projParams.x;
  let far = uAo.projParams.y;
  // WebGPU clip Z is [0, 1] directly -> linear depth = (n * f) / (f - d * (f - n))
  return (near * far) / max(far - d * (far - near), 1e-4);
}

@compute @workgroup_size(8, 8)
fn csMain(@builtin(global_invocation_id) gid: vec3u) {
  let outDim = textureDimensions(uAoOut);
  if (gid.x >= outDim.x || gid.y >= outDim.y) { return; }

  let depthDim = vec2i(textureDimensions(uDepth));
  let maxP = depthDim - vec2i(1);
  let uv = (vec2f(gid.xy) + vec2f(0.5)) / vec2f(outDim);
  let p0 = clamp(vec2i(uv * vec2f(depthDim)), vec2i(1), max(maxP - vec2i(1), vec2i(1)));

  let z0 = linDepthTexel(p0, maxP);
  if (z0 > 240.0) {
    textureStore(uAoOut, vec2i(gid.xy), vec4f(1.0));
    return;
  }

  let worldPerPx = max(z0 * uAo.projParams.w * 2.0 * uAo.fullTexel.y, 1e-4);
  let bias = 0.018 + 0.0015 * z0 + 0.00015 * z0 * z0;
  let angle = fract(52.9829189 * fract(dot(vec2f(gid.xy), vec2f(0.06711056, 0.00583715)))) * 3.14159265;
  let radiusPx = clamp(48.0 / max(z0, 1.5), 3.0, 24.0);

  var occ = 0.0;
  let n = max(uAo.samples / 2u, 4u);
  for (var i = 0u; i < 6u; i = i + 1u) {
    if (i >= n) { break; }
    let a = angle + f32(i) * (3.14159265 / f32(n));
    let rad = radiusPx * (f32(i) + 0.5) / f32(n);
    var dp = vec2i(round(vec2f(cos(a), sin(a)) * rad));
    if (dp.x == 0 && dp.y == 0) { dp = vec2i(1, 0); }
    let zPos = linDepthTexel(p0 + dp, maxP);
    let zNeg = linDepthTexel(p0 - dp, maxP);
    let maxDiff = max(abs(zPos - z0), abs(zNeg - z0));
    let rangeFade = smoothstep(2.2, 0.25, maxDiff);
    let dz = z0 - 0.5 * (zPos + zNeg);
    let armWorld = length(vec2f(dp)) * worldPerPx;
    let crease = clamp((dz - bias) / max(armWorld, 0.04), 0.0, 1.0);
    occ += crease * rangeFade;
  }

  let ao = clamp(1.0 - (occ / f32(n)) * 1.45, 0.48, 1.0);
  textureStore(uAoOut, vec2i(gid.xy), vec4f(ao, ao, ao, 1.0));
}
`;
}

// =========================================================================
// Compute Bloom Extract & Separable Gaussian Blur
// =========================================================================

export const WGSL_BLOOM_EXTRACT_COMPUTE = /* wgsl */ `
struct BloomExtUniforms {
  fullTexel: vec2f,
  _pad: vec2f,
};

@group(0) @binding(0) var<uniform> uExt: BloomExtUniforms;
@group(0) @binding(1) var uSrc: texture_2d<f32>;
@group(0) @binding(2) var uLinSampler: sampler;
@group(0) @binding(3) var uDst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn csMain(@builtin(global_invocation_id) gid: vec3u) {
  let dim = textureDimensions(uDst);
  if (gid.x >= dim.x || gid.y >= dim.y) { return; }
  let uv = (vec2f(gid.xy) + vec2f(0.5)) / vec2f(dim);
  let t = uExt.fullTexel;
  let c = (
    textureSampleLevel(uSrc, uLinSampler, uv + vec2f(-1.0, -1.0) * t, 0.0).rgb +
    textureSampleLevel(uSrc, uLinSampler, uv + vec2f( 1.0, -1.0) * t, 0.0).rgb +
    textureSampleLevel(uSrc, uLinSampler, uv + vec2f(-1.0,  1.0) * t, 0.0).rgb +
    textureSampleLevel(uSrc, uLinSampler, uv + vec2f( 1.0,  1.0) * t, 0.0).rgb
  ) * 0.25;
  let lum = max(c.r, max(c.g, c.b));
  let soft = smoothstep(0.95, 1.45, lum);
  textureStore(uDst, vec2i(gid.xy), vec4f(c * soft, 1.0));
}
`;

export const WGSL_BLOOM_BLUR_COMPUTE = /* wgsl */ `
struct BlurUniforms {
  dir: vec2f,
  _pad: vec2f,
};

@group(0) @binding(0) var<uniform> uBlur: BlurUniforms;
@group(0) @binding(1) var uSrc: texture_2d<f32>;
@group(0) @binding(2) var uLinSampler: sampler;
@group(0) @binding(3) var uDst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn csMain(@builtin(global_invocation_id) gid: vec3u) {
  let dim = textureDimensions(uDst);
  if (gid.x >= dim.x || gid.y >= dim.y) { return; }
  let uv = (vec2f(gid.xy) + vec2f(0.5)) / vec2f(dim);
  let d = uBlur.dir;
  var c = textureSampleLevel(uSrc, uLinSampler, uv, 0.0).rgb * 0.227027;
  c += textureSampleLevel(uSrc, uLinSampler, uv + d * 1.384615, 0.0).rgb * 0.316216;
  c += textureSampleLevel(uSrc, uLinSampler, uv - d * 1.384615, 0.0).rgb * 0.316216;
  c += textureSampleLevel(uSrc, uLinSampler, uv + d * 3.230769, 0.0).rgb * 0.070270;
  c += textureSampleLevel(uSrc, uLinSampler, uv - d * 3.230769, 0.0).rgb * 0.070270;
  textureStore(uDst, vec2i(gid.xy), vec4f(c, 1.0));
}
`;

// =========================================================================
// Fullscreen Composite (AO Denoise + Bloom + Stephen Hill AP1 ACES + sRGB)
// =========================================================================

export const WGSL_COMPOSITE = /* wgsl */ `
${WGSL_ACES}

struct CompositeUniforms {
  aoTexel: vec2f,
  bloomStrength: f32,
  aoEnabled: f32,
  exposure: f32,
  _p0: f32,
  _p1: f32,
  _p2: f32,
};

@group(0) @binding(0) var<uniform> uComp: CompositeUniforms;
@group(0) @binding(1) var uScene: texture_2d<f32>;
@group(0) @binding(2) var uBloom: texture_2d<f32>;
@group(0) @binding(3) var uAo: texture_2d<f32>;
@group(0) @binding(4) var uLinSampler: sampler;

struct VertOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vsMain(@builtin(vertex_index) vid: u32) -> VertOut {
  let x = select(-1.0, 3.0, vid == 1u);
  let y = select(-1.0, 3.0, vid == 2u);
  var out: VertOut;
  out.pos = vec4f(x, y, 0.0, 1.0);
  // WebGPU texture v=0 is top (p.y=+1), v=1 is bottom (p.y=-1)
  out.uv = vec2f(x * 0.5 + 0.5, 0.5 - y * 0.5);
  return out;
}

@fragment
fn fsMain(inp: VertOut) -> @location(0) vec4f {
  var hdr = textureSample(uScene, uLinSampler, inp.uv).rgb;
  let aoSample = (
    textureSample(uAo, uLinSampler, inp.uv + vec2f(-0.5, -0.5) * uComp.aoTexel).r +
    textureSample(uAo, uLinSampler, inp.uv + vec2f( 0.5, -0.5) * uComp.aoTexel).r +
    textureSample(uAo, uLinSampler, inp.uv + vec2f(-0.5,  0.5) * uComp.aoTexel).r +
    textureSample(uAo, uLinSampler, inp.uv + vec2f( 0.5,  0.5) * uComp.aoTexel).r
  ) * 0.25;
  if (uComp.aoEnabled > 0.5) {
    hdr *= mix(1.0, aoSample, 0.85);
  }
  hdr += textureSample(uBloom, uLinSampler, inp.uv).rgb * uComp.bloomStrength;
  let mapped = linearToSrgb(acesToneMap(hdr * uComp.exposure));
  return vec4f(mapped, 1.0);
}
`;

// =========================================================================
// GPU Mipmap Generator (Linear & Gamma-Aware sRGB Blit)
// =========================================================================

export const WGSL_MIPMAP = /* wgsl */ `
@group(0) @binding(0) var uSrc: texture_2d<f32>;
@group(0) @binding(1) var uLinSampler: sampler;

struct VertOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vsMain(@builtin(vertex_index) vid: u32) -> VertOut {
  let x = select(-1.0, 3.0, vid == 1u);
  let y = select(-1.0, 3.0, vid == 2u);
  var out: VertOut;
  out.pos = vec4f(x, y, 0.0, 1.0);
  out.uv = vec2f(x * 0.5 + 0.5, 0.5 - y * 0.5);
  return out;
}

@fragment
fn fsLinear(inp: VertOut) -> @location(0) vec4f {
  return textureSample(uSrc, uLinSampler, inp.uv);
}

fn linToSrgb4(c: vec4f) -> vec4f {
  let lo = c.rgb * 12.92;
  let hi = 1.055 * pow(max(c.rgb, vec3f(0.0)), vec3f(1.0 / 2.4)) - vec3f(0.055);
  return vec4f(select(lo, hi, c.rgb >= vec3f(0.0031308)), c.a);
}

// Samples rgba8unorm-srgb view (hardware linearizes & averages!) and re-encodes to rgba8unorm storage
@fragment
fn fsSrgb(inp: VertOut) -> @location(0) vec4f {
  return linToSrgb4(textureSample(uSrc, uLinSampler, inp.uv));
}
`;
