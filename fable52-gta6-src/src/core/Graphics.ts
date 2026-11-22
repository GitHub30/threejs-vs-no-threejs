import * as THREE from '../engine';
import { setModelDetail, ModelDetail } from './geo';
import type { Environment } from '../world/Environment';
import type { City } from '../world/City';

export type GraphicsLevel = 'low' | 'high' | 'ultra';
export const GRAPHICS_LEVELS: GraphicsLevel[] = ['low', 'high', 'ultra'];
export const GRAPHICS_LABEL: Record<GraphicsLevel, string> = { low: 'ローポリ', high: '高精細', ultra: '最高' };

interface Preset {
  detail: ModelDetail; // model tessellation (see geo.ts)
  pixelRatio: number; // upper bound; the display's own ratio still caps it
  shadow: number; // sun shadow map size (per cascade when there are cascades)
  cascades: number[] | null; // far edge of each shadow cascade in metres; null = one map around the player
  post: boolean; // HDR pipeline: bloom + ACES output
  ao: number; // screen-space depth contact AO scale, 0 = off (needs post)
  reflections: boolean; // analytical sky IBL reflected in paint and metal
}
const PRESETS: Record<GraphicsLevel, Preset> = {
  low: { detail: 0, pixelRatio: 1, shadow: 1024, cascades: null, post: false, ao: 0, reflections: false },
  high: { detail: 1, pixelRatio: 1.25, shadow: 1536, cascades: [24, 85, 280], post: true, ao: 0.5, reflections: true },
  ultra: { detail: 2, pixelRatio: 2, shadow: 2048, cascades: [16, 52, 170, 520], post: true, ao: 0.75, reflections: true },
};

/** Resolution scale steps used when the frame rate cannot be held. */
const SCALE_STEPS = [1, 0.85, 0.7, 0.55];
const KEY = 'velabay.graphics'; // kept so existing settings survive the rename

/** Graphics quality: presets, native CSM shadows, analytical sky IBL, fast HDR post-FX and adaptive resolution. */
export class Graphics {
  level: GraphicsLevel;
  private step = 0;
  private w = 1;
  private h = 1;

  constructor(
    private renderer: THREE.WebGLRenderer,
    private scene: THREE.Scene,
    private camera: THREE.PerspectiveCamera,
    private env: Environment,
    private city: City,
  ) {
    let saved: string | null = null;
    try {
      saved = localStorage.getItem(KEY);
    } catch {
      /* storage unavailable */
    }
    this.level = GRAPHICS_LEVELS.includes(saved as GraphicsLevel) ? (saved as GraphicsLevel) : 'high';
    this.apply(this.level, false);
  }

  private get preset() {
    return PRESETS[this.level];
  }
  get pixelRatio() {
    return Math.min(window.devicePixelRatio, this.preset.pixelRatio) * SCALE_STEPS[this.step];
  }

  apply(level: GraphicsLevel, persist = true) {
    this.level = level;
    this.step = 0;
    const p = this.preset;
    if (persist)
      try {
        localStorage.setItem(KEY, level);
      } catch {
        /* storage unavailable */
      }
    setModelDetail(p.detail);
    this.city.setDetail(p.detail);

    // Configure native WebGL2 TEXTURE_2D_ARRAY CSM shadows (zero shader patching, zero scene traversal)
    this.env.sun.shadow.mapSize.set(p.shadow, p.shadow);
    this.renderer.shadowConfig = {
      enabled: true,
      mapSize: p.shadow,
      cascades: p.cascades,
    };

    // Configure native HDR post-FX + analytical sky IBL (zero PMREM cubemap stalls)
    this.renderer.postConfig = {
      enabled: p.post,
      aoScale: p.post ? p.ao : 0,
      bloomStrength: 0.32,
      reflections: p.reflections,
    };
    this.scene.environmentIntensity = p.reflections ? 0.5 : 0;

    this.resize(this.w, this.h);
  }

  resize(w: number, h: number) {
    this.w = w;
    this.h = h;
    this.renderer.setPixelRatio(this.pixelRatio);
    this.renderer.setSize(w, h, false);
  }

  /** Drop the render resolution one step (the frame rate is too low). Returns false at the floor. */
  stepDown() {
    if (this.step >= SCALE_STEPS.length - 1) return false;
    this.step++;
    this.resize(this.w, this.h);
    return true;
  }

  render(_dt: number) {
    this.renderer.render(this.scene, this.camera);
  }
}
