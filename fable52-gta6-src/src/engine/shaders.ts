// Hand-crafted GLSL ES 3.00 shaders for the custom Osaka City WebGL2 engine.
// Fixed vertex attribute locations shared across main and shadow VAOs:
//   0: aPosition (vec3)   1: aNormal (vec3)    2: aUv (vec2)       3: aUv1 (vec2)
//   4: aColor (vec4)      5: aSkinIndex (vec4) 6: aSkinWeight (vec4)
//   7..10: aInstCol0..3 (mat4)                 11: aInstColor (vec3)

export const FLAG_VERTEX_COLORS   = 1 << 0;
export const FLAG_CAR_SHELL       = 1 << 1;
export const FLAG_PERSON_OUTFIT   = 1 << 2;
export const FLAG_CRM_MAP         = 1 << 3;
export const FLAG_BUMP_MAP        = 1 << 4;
export const FLAG_AO_MAP          = 1 << 5;
export const FLAG_FLAT_SHADING    = 1 << 6;
export const FLAG_FOG             = 1 << 7;
export const FLAG_RECEIVE_SHADOW  = 1 << 8;
export const FLAG_UNLIT           = 1 << 9;
export const FLAG_TONEMAP_INLINE  = 1 << 10;
export const FLAG_BUMP_AO_UV1     = 1 << 11;
export const FLAG_EMISSIVE_MAP    = 1 << 12;
export const FLAG_ALPHA_TO_COV    = 1 << 13;

export const MAX_BONES = 24;
export const MAX_CASCADES = 4;

export type DrawMode = 'static' | 'instanced' | 'skinned';

// Stephen Hill AP1 ACES Filmic fit matching Three.js (divide by 0.6 to cancel Hill's 0.6 exposure factor)
const ACES_GLSL = /* glsl */ `
vec3 rrtAndOdtFit(vec3 v) {
  vec3 a = v * (v + 0.0245786) - 0.000090537;
  vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
  return a / b;
}
vec3 acesToneMap(vec3 color) {
  const mat3 ACESInputMat = mat3(
    0.59719, 0.07600, 0.02840,
    0.35458, 0.90834, 0.13383,
    0.04823, 0.01566, 0.83777
  );
  const mat3 ACESOutputMat = mat3(
     1.60475, -0.10208, -0.00327,
    -0.53108,  1.10813, -0.07276,
    -0.07367, -0.00605,  1.07602
  );
  color /= 0.6;
  color = ACESInputMat * color;
  color = rrtAndOdtFit(color);
  color = ACESOutputMat * color;
  return clamp(color, 0.0, 1.0);
}
vec3 linearToSrgb(vec3 c) {
  vec3 lo = c * 12.92;
  vec3 hi = 1.055 * pow(max(c, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
  return mix(lo, hi, step(vec3(0.0031308), c));
}
`;

export function buildPbrVert(mode: DrawMode): string {
  const isInst = mode === 'instanced';
  const isSkin = mode === 'skinned';
  return /* glsl */ `#version 300 es
precision highp float;

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUv;
layout(location = 3) in vec2 aUv1;
layout(location = 4) in vec4 aColor;
${isSkin ? `
layout(location = 5) in vec4 aSkinIndex;
layout(location = 6) in vec4 aSkinWeight;
uniform mat4 uBones[${MAX_BONES}];
` : ''}
${isInst ? `
layout(location = 7) in vec4 aInstCol0;
layout(location = 8) in vec4 aInstCol1;
layout(location = 9) in vec4 aInstCol2;
layout(location = 10) in vec4 aInstCol3;
layout(location = 11) in vec3 aInstColor;
` : ''}

uniform mat4 uViewProj;
uniform mat4 uView;
uniform mat4 uModel;
uniform vec4 uUvTransform; // xy = repeat, zw = offset

out vec3 vWorldPos;
out vec3 vNormal;
out vec2 vUv;
out vec2 vUv1;
out vec4 vColor;
out float vViewDepth;

void main() {
  vec4 localPos = vec4(aPosition, 1.0);
  vec3 localNor = aNormal;
${isSkin ? `
  ivec4 bi = ivec4(aSkinIndex);
  mat4 skinMat =
    uBones[bi.x] * aSkinWeight.x +
    uBones[bi.y] * aSkinWeight.y +
    uBones[bi.z] * aSkinWeight.z +
    uBones[bi.w] * aSkinWeight.w;
  localPos = skinMat * localPos;
  localNor = mat3(skinMat) * localNor;
` : ''}
${isInst ? `
  mat4 worldMat = uModel * mat4(aInstCol0, aInstCol1, aInstCol2, aInstCol3);
` : `
  mat4 worldMat = uModel;
`}
  vec4 wp = worldMat * localPos;
  vWorldPos = wp.xyz;
  mat3 nm = mat3(worldMat);
  // Inverse-transpose for orthogonal rotation + non-uniform scale (R * S^-1 = nm[i] / |nm[i]|^2), safe at zero-scale bones
  vec3 sx = vec3(dot(nm[0], nm[0]), dot(nm[1], nm[1]), dot(nm[2], nm[2]));
  vec3 invS2 = 1.0 / max(sx, vec3(1e-8));
  mat3 norMat = mat3(nm[0] * invS2.x, nm[1] * invS2.y, nm[2] * invS2.z);
  vec3 wn = norMat * localNor;
  float wnl = length(wn);
  vNormal = wnl > 1e-6 ? wn / wnl : vec3(0.0, 1.0, 0.0);

  vUv = aUv * uUvTransform.xy + uUvTransform.zw;
  vUv1 = aUv1;
  vColor = aColor${isInst ? ' * vec4(aInstColor, 1.0)' : ''};

  vec4 vp = uView * wp;
  vViewDepth = -vp.z;
  gl_Position = uViewProj * wp;
}
`;
}

export const PBR_FRAG = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp sampler2DArrayShadow;

in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vUv;
in vec2 vUv1;
in vec4 vColor;
in float vViewDepth;

// Material uniforms
uniform int uFlags;
uniform vec4 uBaseColor;       // rgb = linear color, a = opacity
uniform vec4 uEmissive;        // rgb = emissive * intensity, a = alphaTest
uniform vec4 uPbrParams;       // x = roughness, y = metalness, z = clearcoat, w = clearcoatRoughness
uniform vec4 uDetailParams;    // x = bumpScale, y = aoIntensity, z = envIntensity, w = exposure

// Texture units (bound to 1x1 defaults when unused -> zero shader recompilation)
uniform sampler2D uMap;           // unit 0
uniform sampler2D uEmissiveMap;   // unit 1
uniform sampler2D uCrmMap;        // unit 2 (R=clearcoat, G=roughness, B=metalness)
uniform sampler2D uBumpAoMap;     // unit 3 (grayscale grain for bump + ao)
uniform sampler2D uPersonMask;    // unit 4
uniform sampler2D uPersonOver;    // unit 5
uniform sampler2DArrayShadow uShadowMap; // unit 6

// People outfit recoloring uniforms
uniform vec4 uPersonTop;
uniform vec4 uPersonBottom;
uniform vec2 uPersonRef;

// Scene lighting uniforms
uniform vec3 uCamPos;
uniform vec3 uSunDir;          // normalized direction TOWARD the sun
uniform vec3 uSunColor;        // linear color * intensity
uniform vec3 uHemiSky;         // linear color * intensity
uniform vec3 uHemiGround;      // linear color * intensity
uniform vec3 uSkyTop;          // analytical IBL zenith color
uniform vec3 uSkyHorizon;      // analytical IBL horizon color
uniform vec4 uFogColorNear;    // rgb = linear fog color, w = fog near
uniform float uFogFar;

// Dynamic local lights (player car headlight + explosion blast light)
uniform vec4 uSpotPosRange;    // xyz = pos, w = distance
uniform vec4 uSpotDirCos;      // xyz = normalized forward dir, w = cos(angle)
uniform vec4 uSpotColorPen;    // rgb = color * intensity, w = penumbra cos inner
uniform float uSpotDecay;
uniform vec4 uPointPosRange;   // xyz = pos, w = distance
uniform vec4 uPointColorDecay; // rgb = color * intensity, w = decay

// Cascaded Shadow Map uniforms
uniform int uCascadeCount;
uniform vec4 uCascadeSplits;      // far depth of cascade 0..3
uniform vec4 uShadowBiases;       // normalBias per cascade
uniform vec4 uShadowDepthBiases;  // NDC depth bias per cascade
uniform mat4 uShadowMats[${MAX_CASCADES}];

out vec4 fragColor;

${ACES_GLSL}

const float PI = 3.14159265;

float distGGX(float NdotH, float rough) {
  float a = rough * rough;
  float a2 = a * a;
  float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
  return a2 / max(PI * d * d, 1e-5);
}

float geomSmith(float NdotV, float NdotL, float rough) {
  float r = rough + 1.0;
  float k = (r * r) * 0.125;
  float gv = NdotV / (NdotV * (1.0 - k) + k);
  float gl = NdotL / (NdotL * (1.0 - k) + k);
  return gv * gl;
}

vec3 fresnelSchlick(float cosT, vec3 F0) {
  float t = clamp(1.0 - cosT, 0.0, 1.0);
  float t2 = t * t;
  return F0 + (1.0 - F0) * (t2 * t2 * t);
}

vec3 fresnelSchlickRoughness(float cosT, vec3 F0, float rough) {
  float t = clamp(1.0 - cosT, 0.0, 1.0);
  float t2 = t * t;
  return F0 + (max(vec3(1.0 - rough), F0) - F0) * (t2 * t2 * t);
}

// Physical Three.js distance attenuation: 1 / max(d^decay, 0.01) * smooth range cutoff
float physDistanceAtten(float dist, float range, float decay) {
  float atten = 1.0 / max(pow(dist, max(decay, 0.01)), 0.01);
  if (range > 0.0) {
    float r = dist / range;
    float r2 = r * r;
    float win = clamp(1.0 - r2 * r2, 0.0, 1.0);
    atten *= win * win;
  }
  return atten;
}

vec3 evalDirectPBR(vec3 N, vec3 V, vec3 L, float NdotV, float NdotL, vec3 albedo, vec3 F0, float rough, float metal, float clearcoat, float ccRough, vec3 radiance) {
  vec3 H = normalize(V + L);
  float NdotH = max(dot(N, H), 0.0);
  float HdotV = max(dot(H, V), 0.0);
  vec3 F = fresnelSchlick(HdotV, F0);
  float D = distGGX(NdotH, rough);
  float G = geomSmith(NdotV, NdotL, rough);
  vec3 spec = (D * G * F) / max(4.0 * NdotV * NdotL, 0.001);
  vec3 kD = (1.0 - F) * (1.0 - metal);
  vec3 outCol = (kD * albedo / PI + spec) * radiance;
  if (clearcoat > 0.01) {
    float ccD = distGGX(NdotH, max(ccRough, 0.05));
    float ccF = 0.04 + 0.96 * pow(1.0 - HdotV, 5.0);
    outCol += vec3(clearcoat * ccD * ccF * 0.25) * radiance;
  }
  return outCol;
}

float sampleCascade(int cIdx, vec3 worldPos, vec3 N, float NdotL) {
  float nBias = uShadowBiases[cIdx] * (1.0 - 0.5 * clamp(NdotL, 0.0, 1.0));
  vec4 sc = uShadowMats[cIdx] * vec4(worldPos + N * nBias, 1.0);
  vec3 proj = sc.xyz / sc.w * 0.5 + 0.5;
  if (proj.x < 0.002 || proj.x > 0.998 || proj.y < 0.002 || proj.y > 0.998 || proj.z < 0.0 || proj.z > 0.999) return 1.0;
  float layer = float(cIdx);
  vec2 texel = 1.0 / vec2(textureSize(uShadowMap, 0).xy);
  float slopeScale = 1.0 + 2.0 * (1.0 - clamp(NdotL, 0.0, 1.0));
  float refZ = proj.z - uShadowDepthBiases[cIdx] * slopeScale;
  float s = 0.0;
  s += texture(uShadowMap, vec4(proj.xy + vec2(-0.7, -0.3) * texel, layer, refZ));
  s += texture(uShadowMap, vec4(proj.xy + vec2( 0.7,  0.3) * texel, layer, refZ));
  s += texture(uShadowMap, vec4(proj.xy + vec2(-0.3,  0.7) * texel, layer, refZ));
  s += texture(uShadowMap, vec4(proj.xy + vec2( 0.3, -0.7) * texel, layer, refZ));
  return s * 0.25;
}

float computeShadow(vec3 worldPos, vec3 N, float NdotL) {
  if (uCascadeCount <= 0) return 1.0;
  int cIdx = uCascadeCount - 1;
  for (int i = 0; i < ${MAX_CASCADES}; i++) {
    if (i >= uCascadeCount) break;
    if (vViewDepth < uCascadeSplits[i]) {
      cIdx = i;
      break;
    }
  }
  if (vViewDepth > uCascadeSplits[uCascadeCount - 1]) return 1.0;
  float sh = sampleCascade(cIdx, worldPos, N, NdotL);
  if (cIdx < uCascadeCount - 1) {
    float splitFar = uCascadeSplits[cIdx];
    float blendStart = splitFar * 0.88;
    if (vViewDepth > blendStart) {
      float f = smoothstep(blendStart, splitFar, vViewDepth);
      sh = mix(sh, sampleCascade(cIdx + 1, worldPos, N, NdotL), f);
    }
  } else {
    float maxFar = uCascadeSplits[uCascadeCount - 1];
    if (vViewDepth > maxFar * 0.88) {
      sh = mix(sh, 1.0, smoothstep(maxFar * 0.88, maxFar, vViewDepth));
    }
  }
  return sh;
}

vec3 evalAnalyticalSkyIBL(vec3 R, float rough) {
  float y = clamp(R.y, -1.0, 1.0);
  vec3 sky = mix(uSkyHorizon, uSkyTop, pow(clamp(y, 0.0, 1.0), 0.55));
  vec3 ground = uHemiGround * 0.45;
  vec3 env = mix(ground, sky, smoothstep(-0.18 - rough * 0.35, 0.18 + rough * 0.35, y));
  float sDot = max(dot(R, uSunDir), 0.0);
  float specPow = mix(96.0, 6.0, rough * rough);
  env += uSunColor * pow(sDot, specPow) * mix(0.28, 0.03, rough);
  return env;
}

void main() {
  int flags = uFlags;
  vec4 texCol = texture(uMap, vUv);
  vec4 diffuse = vec4(uBaseColor.rgb * texCol.rgb, uBaseColor.a * texCol.a);

  bool carLining = false;
  if ((flags & ${FLAG_CAR_SHELL}) != 0) {
    // CarKit shell: centered MSAA alpha-to-coverage ramp across window/arch cutouts while keeping TINT_A=0.6 body panels 100% opaque
    float cov = 1.0;
    if (uEmissive.a > 0.0) {
      if ((flags & ${FLAG_ALPHA_TO_COV}) != 0) {
        cov = clamp((texCol.a - uEmissive.a) / max(fwidth(texCol.a), 1e-4) + 0.5, 0.0, 1.0);
        if (cov == 0.0) discard;
      } else if (texCol.a < uEmissive.a) {
        discard;
      }
    }
    float paintMask = clamp((1.0 - texCol.a) * 2.5, 0.0, 1.0);
    carLining = vColor.a < 0.5;
    diffuse.rgb = carLining ? vColor.rgb : texCol.rgb * mix(vec3(1.0), vColor.rgb, paintMask);
    diffuse.a = cov;
  } else {
    if ((flags & ${FLAG_VERTEX_COLORS}) != 0) {
      // Preserve vertex alpha (critical for CarKit transparent RGBA glass windows!)
      diffuse *= vColor;
    }
    if (uEmissive.a > 0.0) {
      if ((flags & ${FLAG_ALPHA_TO_COV}) != 0) {
        diffuse.a = clamp((diffuse.a - uEmissive.a) / max(fwidth(diffuse.a), 1e-4) + 0.5, 0.0, 1.0);
        if (diffuse.a == 0.0) discard;
      } else if (diffuse.a < uEmissive.a) {
        discard;
      }
    }
  }

  if ((flags & ${FLAG_PERSON_OUTFIT}) != 0) {
    vec4 pm = texture(uPersonMask, vUv);
    float pl = dot(diffuse.rgb, vec3(0.3333));
    diffuse.rgb = mix(diffuse.rgb, uPersonTop.rgb * min(pl / max(uPersonRef.x, 1e-4), 2.2), pm.r * uPersonTop.a);
    diffuse.rgb = mix(diffuse.rgb, uPersonBottom.rgb * min(pl / max(uPersonRef.y, 1e-4), 2.2), pm.g * uPersonBottom.a);
    vec4 po = texture(uPersonOver, vUv);
    diffuse.rgb = mix(diffuse.rgb, po.rgb * (0.86 + 0.3 * pl), po.a);
  }

  vec3 color;
  if ((flags & ${FLAG_UNLIT}) != 0) {
    color = diffuse.rgb;
  } else {
    // Geometric flat normal already faces the camera; only flip smooth vNormal on backfaces
    vec3 N = ((flags & ${FLAG_FLAT_SHADING}) != 0)
      ? normalize(cross(dFdx(vWorldPos), dFdy(vWorldPos)))
      : (gl_FrontFacing ? normalize(vNormal) : -normalize(vNormal));

    vec2 detailUv = ((flags & ${FLAG_BUMP_AO_UV1}) != 0) ? vUv1 : vUv;
    if ((flags & ${FLAG_BUMP_MAP}) != 0) {
      float h = texture(uBumpAoMap, detailUv).r;
      vec3 dp1 = dFdx(vWorldPos);
      vec3 dp2 = dFdy(vWorldPos);
      float dh1 = dFdx(h);
      float dh2 = dFdy(h);
      vec3 r1 = cross(dp2, N);
      vec3 r2 = cross(N, dp1);
      float det = dot(dp1, r1);
      if (abs(det) > 1e-6) {
        vec3 surfGrad = sign(det) * (dh1 * r1 + dh2 * r2);
        N = normalize(abs(det) * N - uDetailParams.x * 0.08 * surfGrad);
      }
    }

    float rough = uPbrParams.x;
    float metal = uPbrParams.y;
    float clearcoat = uPbrParams.z;
    float ccRough = uPbrParams.w;

    if ((flags & ${FLAG_CRM_MAP}) != 0) {
      vec3 crm = texture(uCrmMap, vUv).rgb;
      clearcoat *= crm.r;
      rough *= crm.g;
      metal *= crm.b;
    }
    if (carLining) {
      rough = 0.92;
      metal = 0.0;
      clearcoat = 0.0;
    }
    rough = clamp(rough, 0.06, 1.0);
    metal = clamp(metal, 0.0, 1.0);

    vec3 V = normalize(uCamPos - vWorldPos);
    float NdotV = max(dot(N, V), 0.001);
    vec3 F0 = mix(vec3(0.04), diffuse.rgb, metal);

    // 1. Direct Sun / Moon light with CSM shadows
    vec3 L = uSunDir;
    float NdotL = max(dot(N, L), 0.0);
    float shadow = ((flags & ${FLAG_RECEIVE_SHADOW}) != 0 && NdotL > 0.0)
      ? computeShadow(vWorldPos, N, NdotL)
      : 1.0;

    vec3 Lo = vec3(0.0);
    if (NdotL > 0.0) {
      Lo += evalDirectPBR(N, V, L, NdotV, NdotL, diffuse.rgb, F0, rough, metal, clearcoat, ccRough, uSunColor * (shadow * NdotL));
    }

    // 2. Player car headlight SpotLight (physical 1 / d^decay attenuation + GGX specular)
    if (uSpotPosRange.w > 0.0 && (uSpotColorPen.r + uSpotColorPen.g + uSpotColorPen.b) > 0.001) {
      vec3 toLight = uSpotPosRange.xyz - vWorldPos;
      float dist = length(toLight);
      if (dist < uSpotPosRange.w) {
        vec3 sL = toLight / max(dist, 0.001);
        float cosTheta = dot(-sL, uSpotDirCos.xyz);
        if (cosTheta > uSpotDirCos.w) {
          float spotCone = smoothstep(uSpotDirCos.w, uSpotColorPen.w, cosTheta);
          float atten = physDistanceAtten(dist, uSpotPosRange.w, uSpotDecay);
          float sNdotL = max(dot(N, sL), 0.0);
          if (sNdotL > 0.0) {
            Lo += evalDirectPBR(N, V, sL, NdotV, sNdotL, diffuse.rgb, F0, rough, metal, clearcoat, ccRough, uSpotColorPen.rgb * (spotCone * atten * sNdotL));
          }
        }
      }
    }

    // 3. Explosion blast PointLight (physical 1 / d^decay attenuation + GGX specular)
    if (uPointPosRange.w > 0.0 && (uPointColorDecay.r + uPointColorDecay.g + uPointColorDecay.b) > 0.001) {
      vec3 toP = uPointPosRange.xyz - vWorldPos;
      float dist = length(toP);
      if (dist < uPointPosRange.w) {
        vec3 pL = toP / max(dist, 0.001);
        float atten = physDistanceAtten(dist, uPointPosRange.w, uPointColorDecay.w);
        float pNdotL = max(dot(N, pL), 0.0);
        if (pNdotL > 0.0) {
          Lo += evalDirectPBR(N, V, pL, NdotV, pNdotL, diffuse.rgb, F0, rough, metal, clearcoat, ccRough, uPointColorDecay.rgb * (atten * pNdotL));
        }
      }
    }

    // 4. Hemisphere ambient + Roughness-attenuated Analytical Sky IBL reflection
    float ao = 1.0;
    if ((flags & ${FLAG_AO_MAP}) != 0) {
      ao = mix(1.0, texture(uBumpAoMap, detailUv).r, uDetailParams.y);
    }
    vec3 hemiCol = mix(uHemiGround, uHemiSky, N.y * 0.5 + 0.5);
    vec3 Fenv = fresnelSchlickRoughness(NdotV, F0, rough);
    vec3 kDenv = (1.0 - Fenv) * (1.0 - metal);
    vec3 ambient = kDenv * (diffuse.rgb / PI) * hemiCol * ao;

    if (uDetailParams.z > 0.001) {
      vec3 R = reflect(-V, N);
      vec3 envSpec = evalAnalyticalSkyIBL(R, rough) * Fenv * mix(0.95, 0.12, rough) * uDetailParams.z;
      if (clearcoat > 0.01) {
        float ccFres = 0.04 + 0.96 * pow(1.0 - NdotV, 5.0);
        envSpec += evalAnalyticalSkyIBL(R, ccRough) * (clearcoat * ccFres * uDetailParams.z * 0.95);
      }
      ambient += envSpec * ao;
    }

    // 5. Emissive
    vec3 em = uEmissive.rgb;
    if ((flags & ${FLAG_EMISSIVE_MAP}) != 0) {
      em *= texture(uEmissiveMap, vUv).rgb;
    }

    color = Lo + ambient + em;
  }

  // Distance fog (exact smoothstep matching Three.js THREE.Fog)
  if ((flags & ${FLAG_FOG}) != 0) {
    float fogFactor = smoothstep(uFogColorNear.w, uFogFar, vViewDepth);
    color = mix(color, uFogColorNear.rgb, fogFactor);
  }

  if ((flags & ${FLAG_TONEMAP_INLINE}) != 0) {
    color = linearToSrgb(acesToneMap(color * uDetailParams.w));
  }

  fragColor = vec4(color, diffuse.a);
}
`;

export function buildShadowVert(mode: DrawMode): string {
  const isInst = mode === 'instanced';
  const isSkin = mode === 'skinned';
  return /* glsl */ `#version 300 es
precision highp float;

layout(location = 0) in vec3 aPosition;
layout(location = 2) in vec2 aUv;
${isSkin ? `
layout(location = 5) in vec4 aSkinIndex;
layout(location = 6) in vec4 aSkinWeight;
uniform mat4 uBones[${MAX_BONES}];
` : ''}
${isInst ? `
layout(location = 7) in vec4 aInstCol0;
layout(location = 8) in vec4 aInstCol1;
layout(location = 9) in vec4 aInstCol2;
layout(location = 10) in vec4 aInstCol3;
` : ''}

uniform mat4 uLightVP;
uniform mat4 uModel;
uniform vec4 uUvTransform;

out vec2 vUv;

void main() {
  vec4 localPos = vec4(aPosition, 1.0);
${isSkin ? `
  ivec4 bi = ivec4(aSkinIndex);
  mat4 skinMat =
    uBones[bi.x] * aSkinWeight.x +
    uBones[bi.y] * aSkinWeight.y +
    uBones[bi.z] * aSkinWeight.z +
    uBones[bi.w] * aSkinWeight.w;
  localPos = skinMat * localPos;
` : ''}
${isInst ? `
  mat4 worldMat = uModel * mat4(aInstCol0, aInstCol1, aInstCol2, aInstCol3);
` : `
  mat4 worldMat = uModel;
`}
  vUv = aUv * uUvTransform.xy + uUvTransform.zw;
  gl_Position = uLightVP * (worldMat * localPos);
}
`;
}

export const SHADOW_FRAG = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
in vec2 vUv;
uniform float uAlphaTest;
uniform highp sampler2D uMap;
void main() {
  if (uAlphaTest > 0.0) {
    if (texture(uMap, vUv).a < uAlphaTest) discard;
  }
}
`;

export const SKY_VERT = /* glsl */ `#version 300 es
precision highp float;
layout(location = 0) in vec3 aPosition;
uniform mat4 uViewProj;
uniform mat4 uModel;
out vec3 vDir;
void main() {
  vDir = normalize(aPosition);
  gl_Position = uViewProj * (uModel * vec4(aPosition, 1.0));
}
`;

export const SKY_FRAG = /* glsl */ `#version 300 es
precision highp float;
uniform vec3 top;
uniform vec3 horizon;
uniform vec3 sunDir;
uniform vec3 sunColor;
uniform float stars;
uniform float time;
uniform float cover;
uniform vec3 cloudColor;
uniform float moon;
uniform int uToneMapInline;
uniform float uExposure;

in vec3 vDir;
out vec4 fragColor;

${ACES_GLSL}

float hash(vec3 p){ return fract(sin(dot(p, vec3(12.9898,78.233,37.719))) * 43758.5453); }
float hash2(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453); }
float noise(vec2 p){
  vec2 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(hash2(i), hash2(i + vec2(1.0, 0.0)), f.x),
    mix(hash2(i + vec2(0.0, 1.0)), hash2(i + vec2(1.0, 1.0)), f.x),
    f.y
  );
}
float fbm(vec2 p){
  float a = 0.5, s = 0.0;
  for (int i = 0; i < 5; i++) { s += a * noise(p); p = p * 2.03 + vec2(17.0, 9.0); a *= 0.5; }
  return s;
}

void main() {
  vec3 d = normalize(vDir);
  float h = clamp(d.y, 0.0, 1.0);
  vec3 col = mix(horizon, top, pow(h, 0.55));

  float cloud = 0.0, thick = 0.0;
  if (d.y > 0.0) {
    vec2 uv = d.xz / (d.y + 0.14) * 1.15 + vec2(time * 0.006, time * 0.0025);
    float n = fbm(uv);
    float edge = mix(0.66, 0.26, cover);
    cloud = smoothstep(edge, edge + 0.2, n) * smoothstep(0.0, 0.1, d.y);
    thick = smoothstep(edge + 0.1, edge + 0.45, n);
  }

  float s = max(dot(d, sunDir), 0.0);
  float veil = 1.0 - cloud * 0.92;
  if (moon > 0.5) {
    vec3 r = normalize(cross(sunDir, vec3(0.0, 1.0, 0.0)));
    vec3 u = cross(r, sunDir);
    vec2 q = vec2(dot(d, r), dot(d, u)) / 0.05;
    float disc = smoothstep(0.99870, 0.99895, s);
    float maria = smoothstep(0.42, 0.7, fbm(q * 2.6 + 3.0));
    col += sunColor * (disc * (0.98 - maria * 0.4) + pow(s, 140.0) * 0.12 + pow(s, 10.0) * 0.04) * veil;
  } else {
    float disc = smoothstep(0.99925, 0.99950, s);
    col += sunColor * (disc * 7.0 + pow(s, 260.0) * 1.2 + pow(s, 12.0) * 0.28) * veil;
  }

  vec3 g = floor(d * 220.0);
  float st = step(0.9975, hash(g)) * stars * smoothstep(0.02, 0.3, d.y);
  col += vec3(st) * (1.0 - cloud);

  vec3 cc = cloudColor * (1.0 - thick * 0.38) + sunColor * pow(s, 24.0) * 0.35;
  col = mix(col, cc, cloud * 0.93);

  if (d.y < 0.0) col = mix(horizon, horizon * 0.6, clamp(-d.y * 4.0, 0.0, 1.0));
  if (uToneMapInline != 0) col = linearToSrgb(acesToneMap(col * uExposure));
  fragColor = vec4(col, 1.0);
}
`;

export const POINTS_VERT = /* glsl */ `#version 300 es
precision highp float;
layout(location = 0) in vec3 aPosition;
layout(location = 4) in vec3 aColor;
layout(location = 2) in float aSize;
layout(location = 3) in float aAlpha;
uniform mat4 uViewProj;
uniform mat4 uView;
uniform mat4 uModel;
uniform float uPixelScale;
out vec3 vC;
out float vA;
void main() {
  vC = aColor;
  vA = aAlpha;
  vec4 wp = uModel * vec4(aPosition, 1.0);
  vec4 mv = uView * wp;
  gl_Position = uViewProj * wp;
  gl_PointSize = aSize * 300.0 * uPixelScale / max(1.0, -mv.z);
}
`;

export const POINTS_FRAG = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
in vec3 vC;
in float vA;
uniform int uToneMapInline;
uniform float uExposure;
out vec4 fragColor;
${ACES_GLSL}
void main() {
  float d = length(gl_PointCoord - 0.5);
  if (d > 0.5) discard;
  vec3 c = uToneMapInline != 0 ? linearToSrgb(acesToneMap(vC * uExposure)) : vC;
  fragColor = vec4(c, smoothstep(0.5, 0.15, d) * vA);
}
`;

export const LINES_VERT = /* glsl */ `#version 300 es
precision highp float;
layout(location = 0) in vec3 aPosition;
uniform mat4 uViewProj;
uniform mat4 uModel;
void main() {
  gl_Position = uViewProj * (uModel * vec4(aPosition, 1.0));
}
`;

export const LINES_FRAG = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
uniform vec4 uColor;
uniform int uToneMapInline;
uniform float uExposure;
out vec4 fragColor;
${ACES_GLSL}
void main() {
  vec3 c = uToneMapInline != 0 ? linearToSrgb(acesToneMap(uColor.rgb * uExposure)) : uColor.rgb;
  fragColor = vec4(c, uColor.a);
}
`;

// =========================================================================
// Fast Post-Processing Pipeline (Edge-Preserving Depth AO + Full-Coverage Bloom + ACES)
// =========================================================================

export const FULLSCREEN_VERT = /* glsl */ `#version 300 es
precision highp float;
out vec2 vUv;
void main() {
  vec2 p = vec2((gl_VertexID == 1) ? 3.0 : -1.0, (gl_VertexID == 2) ? 3.0 : -1.0);
  vUv = p * 0.5 + 0.5;
  gl_Position = vec4(p, 0.0, 1.0);
}
`;

/**
 * Derivative-free symmetric paired-sample screen-space contact AO:
 * - Explicit highp sampler2D prevents GLSL ES 3.00 default lowp/fp16 depth truncation on mobile GPUs
 * - Paired opposite samples (+dp, -dp) cancel linear surface slope identically on any flat wall/road/car without 1px normal derivatives
 * - Quadratic depth deadzone guarantees zero contour stripes even on 16-bit mobile depth paths
 */
export const SSAO_FRAG = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
in vec2 vUv;
uniform highp sampler2D uDepth;
uniform vec2 uFullTexel;   // 1.0 / full framebuffer size (consistent radius at any aoScale)
uniform vec4 uProjParams;  // x = near, y = far, z = tanHalfFov * aspect, w = tanHalfFov
uniform int uSamples;
out vec4 fragColor;

float linDepthTexel(ivec2 p, ivec2 maxP) {
  float d = texelFetch(uDepth, clamp(p, ivec2(0), maxP), 0).r;
  float z = d * 2.0 - 1.0;
  float near = uProjParams.x, far = uProjParams.y;
  return (2.0 * near * far) / max(far + near - z * (far - near), 1e-4);
}

void main() {
  ivec2 dim = textureSize(uDepth, 0);
  ivec2 maxP = dim - ivec2(1);
  ivec2 p0 = clamp(ivec2(vUv * vec2(dim)), ivec2(1), max(maxP - ivec2(1), ivec2(1)));
  float z0 = linDepthTexel(p0, maxP);
  if (z0 > 240.0) { fragColor = vec4(1.0); return; }

  float worldPerPx = max(z0 * uProjParams.w * 2.0 * uFullTexel.y, 1e-4);
  // Deadzone rejects planar precision noise (even 16-bit depth) while preserving mid-range crevice AO
  float bias = 0.018 + 0.0015 * z0 + 0.00015 * z0 * z0;

  float angle = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715)))) * 3.14159265;
  float radiusPx = clamp(48.0 / max(z0, 1.5), 3.0, 24.0);
  float occ = 0.0;
  int n = max(uSamples / 2, 4);
  for (int i = 0; i < 6; i++) {
    if (i >= n) break;
    float a = angle + float(i) * (3.14159265 / float(n));
    float rad = radiusPx * (float(i) + 0.5) / float(n);
    ivec2 dp = ivec2(round(vec2(cos(a), sin(a)) * rad));
    if (dp.x == 0 && dp.y == 0) dp = ivec2(1, 0);
    float zPos = linDepthTexel(p0 + dp, maxP);
    float zNeg = linDepthTexel(p0 - dp, maxP);
    float maxDiff = max(abs(zPos - z0), abs(zNeg - z0));
    float rangeFade = smoothstep(2.2, 0.25, maxDiff);
    float dz = z0 - 0.5 * (zPos + zNeg);
    float armWorld = length(vec2(dp)) * worldPerPx;
    float crease = clamp((dz - bias) / max(armWorld, 0.04), 0.0, 1.0);
    occ += crease * rangeFade;
  }
  float ao = clamp(1.0 - (occ / float(n)) * 1.45, 0.48, 1.0);
  fragColor = vec4(ao, ao, ao, 1.0);
}
`;

export const BLOOM_EXTRACT_FRAG = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
in vec2 vUv;
uniform highp sampler2D uSrc;
uniform vec2 uTexel; // 1.0 / full framebuffer size
out vec4 fragColor;
void main() {
  // 4 bilinear taps at +-1.0 full-res texel cover 100% of each 4x4 full-res block (zero strobe flicker!)
  vec3 c = (
    texture(uSrc, vUv + vec2(-1.0, -1.0) * uTexel).rgb +
    texture(uSrc, vUv + vec2( 1.0, -1.0) * uTexel).rgb +
    texture(uSrc, vUv + vec2(-1.0,  1.0) * uTexel).rgb +
    texture(uSrc, vUv + vec2( 1.0,  1.0) * uTexel).rgb
  ) * 0.25;
  float lum = max(c.r, max(c.g, c.b));
  float soft = smoothstep(0.95, 1.45, lum);
  fragColor = vec4(c * soft, 1.0);
}
`;

export const BLOOM_BLUR_FRAG = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
in vec2 vUv;
uniform highp sampler2D uSrc;
uniform vec2 uDir;
out vec4 fragColor;
void main() {
  vec3 c = texture(uSrc, vUv).rgb * 0.227027;
  c += texture(uSrc, vUv + uDir * 1.384615).rgb * 0.316216;
  c += texture(uSrc, vUv - uDir * 1.384615).rgb * 0.316216;
  c += texture(uSrc, vUv + uDir * 3.230769).rgb * 0.070270;
  c += texture(uSrc, vUv - uDir * 3.230769).rgb * 0.070270;
  fragColor = vec4(c, 1.0);
}
`;

export const COMPOSITE_FRAG = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
in vec2 vUv;
uniform highp sampler2D uScene;
uniform highp sampler2D uBloom;
uniform highp sampler2D uAo;
uniform vec2 uAoTexel;
uniform float uBloomStrength;
uniform float uAoEnabled;
uniform float uExposure;
out vec4 fragColor;
${ACES_GLSL}
void main() {
  vec3 hdr = texture(uScene, vUv).rgb;
  if (uAoEnabled > 0.5) {
    // 4-tap rotated-grid bilinear denoise smooths interleaved-noise AO in zero extra passes
    float ao = (
      texture(uAo, vUv + vec2(-0.5, -0.5) * uAoTexel).r +
      texture(uAo, vUv + vec2( 0.5, -0.5) * uAoTexel).r +
      texture(uAo, vUv + vec2(-0.5,  0.5) * uAoTexel).r +
      texture(uAo, vUv + vec2( 0.5,  0.5) * uAoTexel).r
    ) * 0.25;
    hdr *= mix(1.0, ao, 0.85);
  }
  hdr += texture(uBloom, vUv).rgb * uBloomStrength;
  vec3 mapped = linearToSrgb(acesToneMap(hdr * uExposure));
  fragColor = vec4(mapped, 1.0);
}
`;
