// Unified async renderer factory: prefers WebGPU and falls back seamlessly to WebGL2.
// Supports URL query override (?backend=webgpu | ?backend=webgl2) for live A/B comparison.

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
  if (pref !== 'webgl2') {
    try {
      return await WebGPURenderer.create(opts, pref === 'webgpu');
    } catch (err) {
      console.info('[Engine] WebGPU unavailable, falling back to WebGL2:', err instanceof Error ? err.message : err);
    }
  }
  return new WebGLRenderer(opts);
}
