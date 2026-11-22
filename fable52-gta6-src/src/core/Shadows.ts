import * as THREE from '../engine';

/**
 * Cascaded shadow map configuration wrapper.
 * In the custom engine, CSM runs natively via hardware WebGL2 TEXTURE_2D_ARRAY shadows inside WebGLRenderer,
 * eliminating per-frame scene.traverse() and shader regex patching.
 */
export class SunShadows {
  constructor(
    private renderer: THREE.WebGLRenderer,
    splits: number[],
    mapSize: number,
  ) {
    this.renderer.shadowConfig = { enabled: true, mapSize, cascades: splits };
  }

  update() {
    // Handled natively inside WebGLRenderer.render()
  }

  dispose() {
    this.renderer.shadowConfig = { enabled: true, mapSize: 1024, cascades: null };
  }
}
