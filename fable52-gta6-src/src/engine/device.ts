// Device & memory budget detection (protects 3 GB iOS Safari / low-memory mobile from Jetsam OOM).

function queryFlag(name: string): boolean {
  try {
    return new URLSearchParams(window.location.search).get(name) === '1';
  } catch {
    return false;
  }
}

const ua = typeof navigator !== 'undefined' ? navigator.userAgent : '';
const platform = typeof navigator !== 'undefined' ? navigator.platform : '';
const touchPoints = typeof navigator !== 'undefined' ? navigator.maxTouchPoints || 0 : 0;
const deviceMem = typeof navigator !== 'undefined'
  ? (navigator as unknown as { deviceMemory?: number }).deviceMemory
  : undefined;

/** True on iPhone, iPad, and iPod (including iPadOS desktop-class Safari UA). */
export const IS_IOS = /iPhone|iPad|iPod/i.test(ua) || (platform === 'MacIntel' && touchPoints > 1);

/** True on mobile phones/tablets (iOS or Android). */
export const IS_MOBILE = IS_IOS || /Android|Mobile/i.test(ua);

/**
 * True on memory-constrained devices (all iOS devices with Safari's strict ~224-384 MB Canvas2D
 * and ~600 MB Jetsam tab limits, or devices reporting <= 4 GB RAM). Also supports ?lowmem=1 testing.
 */
export const LOW_MEM_DEVICE = IS_IOS || (deviceMem !== undefined && deviceMem <= 4) || queryFlag('lowmem');

/** Maximum single texture edge (px) uploaded to GPU on this device. */
export const MAX_TEX_DIM = LOW_MEM_DEVICE ? 1024 : 8192;
