// Zero-dependency client for collecting rich device/User-Agent metadata and syncing
// benchmark results automatically with the Cloudflare Worker + D1 telemetry API.

import { supportsWasmSimd } from '../custom-engine/wasm-bridge.js';

export const APP_VERSION = '1.1.0';
export const TELEMETRY_API_BASE = 'https://threejs-vs-no-threejs-telemetry.6u.workers.dev';

let cachedWebGLInfo = null;
let cachedWebGPUInfo = null;

function probeWebGLInfo() {
  if (cachedWebGLInfo) return cachedWebGLInfo;
  const info = {
    renderer: null,
    vendor: null,
    unmaskedRenderer: null,
    unmaskedVendor: null,
  };
  try {
    const canvas = document.createElement('canvas');
    canvas.width = 1;
    canvas.height = 1;
    const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
    if (gl) {
      info.renderer = gl.getParameter(gl.RENDERER) || null;
      info.vendor = gl.getParameter(gl.VENDOR) || null;
      const dbg = gl.getExtension('WEBGL_debug_renderer_info');
      if (dbg) {
        info.unmaskedRenderer = gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) || null;
        info.unmaskedVendor = gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL) || null;
      }
      gl.getExtension('WEBGL_lose_context')?.loseContext();
    }
  } catch {
    // Ignore restricted environments
  }
  cachedWebGLInfo = info;
  return info;
}

async function probeWebGPUInfo() {
  if (cachedWebGPUInfo) return cachedWebGPUInfo;
  const info = {
    supported: !!navigator.gpu,
    preferredFormat: null,
    adapterInfo: null,
    hasTimestampQuery: false,
    limits: null,
  };
  if (navigator.gpu) {
    try {
      info.preferredFormat = navigator.gpu.getPreferredCanvasFormat?.() || null;
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (adapter) {
        const ai = adapter.info || (typeof adapter.requestAdapterInfo === 'function' ? await adapter.requestAdapterInfo().catch(() => null) : null);
        info.adapterInfo = ai
          ? {
              vendor: ai.vendor || null,
              architecture: ai.architecture || null,
              device: ai.device || null,
              description: ai.description || null,
              isFallbackAdapter: typeof adapter.isFallbackAdapter === 'boolean' ? adapter.isFallbackAdapter : null,
            }
          : null;
        info.hasTimestampQuery = adapter.features?.has('timestamp-query') || false;
        info.limits = adapter.limits
          ? {
              maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
              maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
              maxBufferSize: adapter.limits.maxBufferSize,
            }
          : null;
      }
    } catch {
      // Ignore adapter errors
    }
  }
  cachedWebGPUInfo = info;
  return info;
}

async function probeUserAgentData() {
  const uad = navigator.userAgentData;
  if (!uad) return null;
  const out = {
    brands: Array.isArray(uad.brands) ? uad.brands.map((b) => ({ brand: b.brand, version: b.version })) : null,
    mobile: typeof uad.mobile === 'boolean' ? uad.mobile : null,
    platform: uad.platform || null,
    architecture: null,
    bitness: null,
    model: null,
    platformVersion: null,
    fullVersionList: null,
  };
  if (typeof uad.getHighEntropyValues === 'function') {
    try {
      const hi = await uad.getHighEntropyValues([
        'architecture',
        'bitness',
        'model',
        'platformVersion',
        'fullVersionList',
      ]);
      out.architecture = hi.architecture || null;
      out.bitness = hi.bitness || null;
      out.model = hi.model || null;
      out.platformVersion = hi.platformVersion || null;
      out.fullVersionList = Array.isArray(hi.fullVersionList)
        ? hi.fullVersionList.map((b) => ({ brand: b.brand, version: b.version }))
        : null;
    } catch {
      // High-entropy UA hints may be restricted by Permissions-Policy
    }
  }
  return out;
}

/**
 * Collects rich non-PII hardware, browser, User-Agent, GPU, display, and fullscreen metadata.
 */
export async function collectDeviceInfo({ customEngine, threeEngine, fullscreenState }) {
  const [userAgentData, webgpu] = await Promise.all([
    probeUserAgentData(),
    probeWebGPUInfo(),
  ]);
  const webgl = probeWebGLInfo();

  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const stage = document.getElementById('stage');
  const vpW = stage?.clientWidth || window.innerWidth;
  const vpH = stage?.clientHeight || window.innerHeight;

  let timezone = null;
  try {
    timezone = Intl.DateTimeFormat().resolvedOptions().timeZone || null;
  } catch {}

  return {
    capturedAt: new Date().toISOString(),
    pageUrl: window.location.href,
    referrer: document.referrer || null,
    userAgent: navigator.userAgent || '',
    userAgentData,
    platform: navigator.platform || null,
    language: navigator.language || null,
    languages: Array.isArray(navigator.languages) ? navigator.languages.slice(0, 5) : null,
    timezone,
    hardwareConcurrency: navigator.hardwareConcurrency || null,
    deviceMemoryGB: typeof navigator.deviceMemory === 'number' ? navigator.deviceMemory : null,
    maxTouchPoints: typeof navigator.maxTouchPoints === 'number' ? navigator.maxTouchPoints : null,
    screen: {
      width: window.screen?.width || null,
      height: window.screen?.height || null,
      availWidth: window.screen?.availWidth || null,
      availHeight: window.screen?.availHeight || null,
      colorDepth: window.screen?.colorDepth || null,
      orientationType: window.screen?.orientation?.type || (vpW >= vpH ? 'landscape-primary' : 'portrait-primary'),
      orientationAngle: typeof window.screen?.orientation?.angle === 'number' ? window.screen.orientation.angle : null,
    },
    viewport: {
      width: Math.round(vpW),
      height: Math.round(vpH),
      rawDpr: window.devicePixelRatio || 1,
      dpr,
    },
    fullscreen: {
      active: !!document.fullscreenElement || !!document.webkitFullscreenElement || !!fullscreenState?.cssLandscapeFallback,
      nativeFullscreen: !!document.fullscreenElement || !!document.webkitFullscreenElement,
      orientationLocked: !!fullscreenState?.orientationLocked,
      cssLandscapeFallback: !!fullscreenState?.cssLandscapeFallback,
    },
    wasm: {
      simd: supportsWasmSimd(),
      moduleLabel: customEngine?.wasmLabel || null,
      wasmBytes: customEngine?.wasm?.byteLength || null,
    },
    webgpu,
    webgl,
    engines: {
      customBackend: customEngine?.backendLabel || null,
      threeBackend: threeEngine?.backendLabel || null,
      customInitMs: typeof customEngine?.initMs === 'number' ? Number(customEngine.initMs.toFixed(2)) : null,
      threeInitMs: typeof threeEngine?.initMs === 'number' ? Number(threeEngine.initMs.toFixed(2)) : null,
    },
  };
}

/**
 * Automatically submits a completed benchmark run to the Cloudflare Worker + D1 telemetry API.
 */
export async function submitBenchmarkResult({ device, results }) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 12000);
  try {
    const res = await fetch(`${TELEMETRY_API_BASE}/api/results`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        appVersion: APP_VERSION,
        device,
        results,
      }),
      keepalive: true,
      signal: controller.signal,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || !data.ok) {
      return { ok: false, error: data.error || `HTTP ${res.status}` };
    }
    return data;
  } catch (err) {
    return { ok: false, error: String(err?.message || err) };
  } finally {
    clearTimeout(timer);
  }
}

export async function fetchRecentResults(limit = 20) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 10000);
  try {
    const res = await fetch(`${TELEMETRY_API_BASE}/api/results?limit=${limit}`, {
      signal: controller.signal,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || !data.ok) {
      return { ok: false, error: data.error || `HTTP ${res.status}`, runs: [], totalRuns: 0 };
    }
    return data;
  } catch (err) {
    return { ok: false, error: String(err?.message || err), runs: [], totalRuns: 0 };
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Formats a concise human-readable GPU / device label from a collected run row.
 */
export function formatRunDeviceLabel(run) {
  const rawGpu = run.gpuDescription || run.webglRenderer || [run.gpuVendor, run.gpuArch].filter(Boolean).join(' ') || 'Unknown GPU';
  const cleanGpu = rawGpu
    .replace(/^ANGLE \(([^,]+),\s*/, '')
    .replace(/\s*\(0x[0-9a-fA-F]+\).*$/, '')
    .replace(/Direct3D11.*$/, '')
    .replace(/OpenGL.*$/, '')
    .replace(/,\s*$/, '')
    .trim();
  const os = run.uaPlatform || (run.userAgent?.includes('Mac') ? 'macOS' : run.userAgent?.includes('Win') ? 'Windows' : run.userAgent?.includes('Android') ? 'Android' : run.userAgent?.includes('iPhone') || run.userAgent?.includes('iPad') ? 'iOS' : 'Linux');
  const model = run.uaModel ? ` · ${run.uaModel}` : '';
  return {
    gpu: cleanGpu.slice(0, 48),
    os: `${os}${run.uaMobile ? ' (Mobile)' : ''}${model}`,
  };
}
