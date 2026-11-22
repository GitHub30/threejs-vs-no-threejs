// Unified async renderer factory: prefers WebGPU and falls back seamlessly to WebGL2.
// Supports URL query override (?backend=webgpu | ?backend=webgl2) for live A/B comparison.

import { IS_IOS } from './device';
import type { EngineRenderer, RendererBackend, RendererOptions } from './gpuTypes';
import { WebGLRenderer } from './renderer';
import { WebGPURenderer } from './wgpuRenderer';

export function resolveRequestedBackend(opts?: RendererOptions): 'auto' | RendererBackend {
  if (opts?.preferredBackend && opts.preferredBackend !== 'auto') return opts.preferredBackend;
  try {
    const q = new URLSearchParams(window.location.search).get('backend')?.toLowerCase();
    if (q === 'webgpu' || q === 'webgl2') return q;
  } catch {
    /* ignore */
  }
  return 'auto';
}

export async function createRenderer(opts: RendererOptions): Promise<EngineRenderer> {
  const pref = resolveRequestedBackend(opts);
  // On iOS Safari, default 'auto' to WebGL2 (ANGLE Metal) to avoid WebKit's GPUProcess memory leak
  // on 3 GB iPhones (WebKit bug #303203); users can still opt into WebGPU via ?backend=webgpu.
  const tryWebGpu = pref === 'webgpu' || (pref === 'auto' && !IS_IOS);
  if (tryWebGpu) {
    try {
      return await WebGPURenderer.create(opts, pref === 'webgpu');
    } catch (err) {
      console.info('[Engine] WebGPU unavailable, falling back to WebGL2:', err instanceof Error ? err.message : err);
    }
  }
  return new WebGLRenderer(opts);
}
