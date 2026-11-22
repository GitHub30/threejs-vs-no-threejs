// Main Application Controller: manages synchronized dual-engine rendering, live HUD telemetry,
// interactive camera & wind-vortex controls, and fair sequential solo automated benchmarking.

import { ENGINE_FOOTPRINT, SCENARIOS } from './shared/config.js';
import { OrbitCamera } from './shared/camera.js';
import {
  drawBenchmarkBarChart,
  drawSparkline,
  FrameTelemetry,
} from './shared/metrics.js';
import { CustomWebGPUEngine } from './custom-engine/webgpu-engine.js';
import { CustomWebGL2Engine } from './custom-engine/webgl2-fallback.js';
import { ThreeJSEngine } from './three-engine/three-engine.js';

const dom = {
  stage: document.getElementById('stage'),
  paneCustom: document.getElementById('pane-custom'),
  paneThree: document.getElementById('pane-three'),
  canvasCustom: document.getElementById('canvas-custom'),
  canvasThree: document.getElementById('canvas-three'),

  // Controls
  scenarioBtns: Array.from(document.querySelectorAll('[data-scenario]')),
  viewBtns: Array.from(document.querySelectorAll('[data-view]')),
  selectCount: document.getElementById('select-count'),
  selectThreeBackend: document.getElementById('select-three-backend'),
  btnRunBench: document.getElementById('btn-run-bench'),
  btnOpenReport: document.getElementById('btn-open-report'),
  btnToggleOrbit: document.getElementById('btn-toggle-orbit'),
  btnResetSim: document.getElementById('btn-reset-sim'),

  // Bottom caption
  captionTitle: document.getElementById('caption-title'),
  captionDesc: document.getElementById('caption-desc'),

  // HUD Custom
  customFps: document.getElementById('custom-fps'),
  customStack: document.getElementById('custom-stack'),
  customFrameMs: document.getElementById('custom-frame-ms'),
  customCpuMs: document.getElementById('custom-cpu-ms'),
  customGpuMs: document.getElementById('custom-gpu-ms'),
  customP99Ms: document.getElementById('custom-p99-ms'),
  customVisible: document.getElementById('custom-visible'),
  customInitMs: document.getElementById('custom-init-ms'),
  sparkCustom: document.getElementById('spark-custom'),

  // HUD Three.js
  threeFps: document.getElementById('three-fps'),
  threeStack: document.getElementById('three-stack'),
  threeBackendTag: document.getElementById('three-backend-tag'),
  threeFrameMs: document.getElementById('three-frame-ms'),
  threeCpuMs: document.getElementById('three-cpu-ms'),
  threeGpuMs: document.getElementById('three-gpu-ms'),
  threeP99Ms: document.getElementById('three-p99-ms'),
  threeVisible: document.getElementById('three-visible'),
  threeInitMs: document.getElementById('three-init-ms'),
  sparkThree: document.getElementById('spark-three'),

  // Benchmark Banner & Report Modal
  benchBanner: document.getElementById('bench-banner'),
  benchStatusTitle: document.getElementById('bench-status-title'),
  benchStatusDetail: document.getElementById('bench-status-detail'),
  benchProgressFill: document.getElementById('bench-progress-fill'),
  btnCancelBench: document.getElementById('btn-cancel-bench'),

  reportModal: document.getElementById('report-modal'),
  reportEnvInfo: document.getElementById('report-env-info'),
  reportEmpty: document.getElementById('report-empty'),
  reportContent: document.getElementById('report-content'),
  reportKpis: document.getElementById('report-kpis'),
  reportTbody: document.getElementById('report-tbody'),
  chartCpu: document.getElementById('chart-cpu'),
  chartFps: document.getElementById('chart-fps'),
  btnBenchCurrent: document.getElementById('btn-bench-current'),
  btnBenchAll: document.getElementById('btn-bench-all'),
  btnCloseReport: document.getElementById('btn-close-report'),
};

const state = {
  scenario: 'hybrid',
  viewMode: 'split', // 'split' | 'custom' | 'three'
  entityCount: SCENARIOS.hybrid.defaultCount,
  threeBackend: 'webgpu',
  hasWebGPU: false,

  customEngine: null,
  threeEngine: null,
  customTelemetry: new FrameTelemetry(120),
  threeTelemetry: new FrameTelemetry(120),

  camera: new OrbitCamera(),
  lastRafTime: 0,
  lastHudUpdate: 0,

  benchmarkRunning: false,
  benchmarkCancelled: false,
  benchmarkResults: [], // Array of { scenario, count, custom, three }
};

function populateCountSelect(scenarioId, selectedCount) {
  const sc = SCENARIOS[scenarioId];
  dom.selectCount.innerHTML = '';
  for (const c of sc.counts) {
    const opt = document.createElement('option');
    opt.value = String(c);
    opt.textContent = `${c.toLocaleString()} ${sc.entityUnit}`;
    if (c === selectedCount) opt.selected = true;
    dom.selectCount.appendChild(opt);
  }
  dom.captionTitle.textContent = sc.title;
  dom.captionDesc.textContent = sc.description;
}

function updateViewModeUI(mode) {
  state.viewMode = mode;
  dom.stage.classList.remove('mode-split', 'mode-custom', 'mode-three');
  dom.stage.classList.add(`mode-${mode}`);
  for (const btn of dom.viewBtns) {
    btn.classList.toggle('active', btn.dataset.view === mode);
  }
  handleResize();
}

function syncTopbarOffset() {
  const topbar = document.querySelector('.topbar');
  if (!topbar) return;
  const rect = topbar.getBoundingClientRect();
  document.documentElement.style.setProperty('--hud-top', `${Math.ceil(rect.bottom + 10)}px`);
}

function handleResize() {
  syncTopbarOffset();
  const showCustom = state.viewMode === 'split' || state.viewMode === 'custom';
  const showThree = state.viewMode === 'split' || state.viewMode === 'three';

  if (showCustom && state.customEngine) {
    const w = dom.paneCustom.clientWidth || window.innerWidth;
    const h = dom.paneCustom.clientHeight || window.innerHeight;
    state.customEngine.resize(w, h);
    state.camera.setAspect(w / h);
  }
  if (showThree && state.threeEngine) {
    const w = dom.paneThree.clientWidth || window.innerWidth;
    const h = dom.paneThree.clientHeight || window.innerHeight;
    state.threeEngine.resize(w, h);
    state.camera.setAspect(w / h);
  }
}

async function initCustomEngine() {
  if (state.customEngine) {
    state.customEngine.dispose();
    state.customEngine = null;
  }
  // Replace canvas element cleanly if switching WebGPU <-> WebGL context types
  const webgpuOk = await CustomWebGPUEngine.isAvailable();
  state.hasWebGPU = webgpuOk;

  const engine = webgpuOk
    ? new CustomWebGPUEngine(dom.canvasCustom)
    : new CustomWebGL2Engine(dom.canvasCustom);

  await engine.init(state.scenario, state.entityCount);
  state.customEngine = engine;
  state.customTelemetry.reset();
  state.customTelemetry.initMs = engine.initMs;
  state.customTelemetry.backendLabel = engine.backendLabel;
  state.customTelemetry.wasmLabel = engine.wasmLabel;

  dom.customStack.textContent = `${engine.backendLabel} · ${engine.wasmLabel}`;
  dom.customInitMs.textContent = `${engine.initMs.toFixed(0)} ms`;
}

async function initThreeEngine(backendOverride = null) {
  const preferred = backendOverride || state.threeBackend;
  if (state.threeEngine) {
    state.threeEngine.dispose();
    state.threeEngine = null;
    // Recreate canvas cleanly because a canvas bound to webgpu cannot be reused by webgl2 (and vice versa)
    const fresh = document.createElement('canvas');
    fresh.id = 'canvas-three';
    fresh.className = 'gl-canvas';
    dom.canvasThree.replaceWith(fresh);
    dom.canvasThree = fresh;
  }

  const engine = new ThreeJSEngine(dom.canvasThree, preferred);
  await engine.init(state.scenario, state.entityCount);
  state.threeEngine = engine;
  state.threeBackend = engine.activeBackend;
  dom.selectThreeBackend.value = engine.activeBackend;

  state.threeTelemetry.reset();
  state.threeTelemetry.initMs = engine.initMs;
  state.threeTelemetry.backendLabel = engine.backendLabel;

  dom.threeBackendTag.textContent = engine.activeBackend === 'webgpu' ? 'WebGPU' : 'WebGL2';
  dom.threeStack.textContent = engine._computeFallbackToHybrid
    ? `${engine.backendLabel} · Compute非対応(Hybrid)`
    : `${engine.backendLabel} · JS (Instanced/TSL)`;
  dom.threeInitMs.textContent = `${engine.initMs.toFixed(0)} ms`;
}

function reconfigureEngines(scenarioId, count) {
  state.scenario = scenarioId;
  state.entityCount = count;
  populateCountSelect(scenarioId, count);

  for (const btn of dom.scenarioBtns) {
    btn.classList.toggle('active', btn.dataset.scenario === scenarioId);
  }

  state.customEngine?.configure(scenarioId, count);
  state.threeEngine?.configure(scenarioId, count);
  state.customTelemetry.reset();
  state.threeTelemetry.reset();

  if (state.threeEngine) {
    dom.threeStack.textContent = state.threeEngine._computeFallbackToHybrid
      ? `${state.threeEngine.backendLabel} · Compute非対応(Hybrid)`
      : `${state.threeEngine.backendLabel} · ${scenarioId === 'gpu-compute' ? 'TSL Compute' : 'JS + Scene'}`;
  }
}

function updateHudDom() {
  const showCustom = state.viewMode === 'split' || state.viewMode === 'custom';
  const showThree = state.viewMode === 'split' || state.viewMode === 'three';

  if (showCustom) {
    const s = state.customTelemetry.getLiveSummary();
    dom.customFps.textContent = s.fps > 0 ? Math.round(s.fps).toString() : '--';
    dom.customFrameMs.textContent = `${s.avgFrameMs.toFixed(1)} ms`;
    dom.customCpuMs.textContent = `${s.avgCpuMs.toFixed(2)} ms`;
    dom.customGpuMs.textContent = s.avgGpuMs >= 0 ? `${s.avgGpuMs.toFixed(2)} ms` : 'N/A';
    dom.customP99Ms.textContent = `${s.p99FrameMs.toFixed(1)} ms`;
    dom.customVisible.textContent = s.visibleCount.toLocaleString();
    drawSparkline(dom.sparkCustom, state.customTelemetry, 'rgb(15, 118, 110)');
  }

  if (showThree) {
    const s = state.threeTelemetry.getLiveSummary();
    dom.threeFps.textContent = s.fps > 0 ? Math.round(s.fps).toString() : '--';
    dom.threeFrameMs.textContent = `${s.avgFrameMs.toFixed(1)} ms`;
    dom.threeCpuMs.textContent = `${s.avgCpuMs.toFixed(2)} ms`;
    dom.threeGpuMs.textContent = s.avgGpuMs >= 0 ? `${s.avgGpuMs.toFixed(2)} ms` : 'N/A';
    dom.threeP99Ms.textContent = `${s.p99FrameMs.toFixed(1)} ms`;
    dom.threeVisible.textContent = s.visibleCount.toLocaleString();
    drawSparkline(dom.sparkThree, state.threeTelemetry, 'rgb(194, 65, 12)');
  }
}

function animationLoop(now) {
  requestAnimationFrame(animationLoop);

  if (state.lastRafTime === 0) {
    state.lastRafTime = now;
    return;
  }
  const frameDeltaMs = Math.min(5000, Math.max(0.5, now - state.lastRafTime));
  state.lastRafTime = now;
  const dt = Math.min(0.05, frameDeltaMs * 0.001);

  state.camera.update(dt);

  const showCustom = state.viewMode === 'split' || state.viewMode === 'custom';
  const showThree = state.viewMode === 'split' || state.viewMode === 'three';

  if (showCustom && state.customEngine?.initialized) {
    const res = state.customEngine.render(state.camera, dt);
    state.customTelemetry.record(frameDeltaMs, res.cpuMs, res.gpuMs, res.visibleCount);
  }
  if (showThree && state.threeEngine?.initialized) {
    const res = state.threeEngine.render(state.camera, dt);
    state.threeTelemetry.record(frameDeltaMs, res.cpuMs, res.gpuMs, res.visibleCount);
  }

  if (now - state.lastHudUpdate >= 160) {
    state.lastHudUpdate = now;
    updateHudDom();
  }
}

// --- Fair Automated Sequential Solo Benchmark Harness ---

function waitWarmup(minFrames = 4, maxFrames = 16, maxMs = 800) {
  return new Promise((resolve) => {
    const start = performance.now();
    let count = 0;
    function tick() {
      count++;
      if (count >= maxFrames || (count >= minFrames && performance.now() - start >= maxMs)) {
        resolve();
      } else {
        requestAnimationFrame(tick);
      }
    }
    requestAnimationFrame(tick);
  });
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function measureEngineSolo(engineKey, scenarioId, count, durationMs = 2400) {
  updateViewModeUI(engineKey);
  state.camera.resetForBenchmark();
  reconfigureEngines(scenarioId, count);

  const telemetry = engineKey === 'custom' ? state.customTelemetry : state.threeTelemetry;
  // Adaptive warmup so JIT & GPU pipelines settle on fast GPUs without stalling slow devices
  await waitWarmup(3, 16, 750);
  if (state.benchmarkCancelled) return null;

  telemetry.startRecording();
  const startAt = performance.now();
  const endAt = startAt + durationMs;
  const hardLimit = startAt + Math.max(durationMs * 2.5, 5000);
  while (!state.benchmarkCancelled) {
    const now = performance.now();
    if (now >= endAt && telemetry.recFrames.length >= 4) break;
    if (now >= hardLimit) break;
    await sleep(50);
  }
  const stats = telemetry.stopRecording();
  stats.backend = engineKey === 'custom'
    ? state.customEngine?.backendLabel || 'WebGPU'
    : state.threeEngine?.backendLabel || 'Three.js';
  return stats;
}

async function runAutomatedBenchmark(runAllScenarios = false, durationPerEngineMs = 2400) {
  if (state.benchmarkRunning) return;
  state.benchmarkRunning = true;
  state.benchmarkCancelled = false;

  const prevView = state.viewMode;
  const prevScenario = state.scenario;
  const prevCount = state.entityCount;

  dom.reportModal.classList.add('hidden');
  dom.benchBanner.classList.remove('hidden');

  const plan = runAllScenarios
    ? [
        { scenario: 'hybrid', count: SCENARIOS.hybrid.benchCount },
        { scenario: 'gpu-compute', count: SCENARIOS['gpu-compute'].benchCount },
        { scenario: 'scene-graph', count: SCENARIOS['scene-graph'].benchCount },
      ]
    : [{ scenario: state.scenario, count: state.entityCount }];

  const totalSteps = plan.length * 2;
  let step = 0;
  const results = [];

  try {
    for (const item of plan) {
      if (state.benchmarkCancelled) break;
      const sc = SCENARIOS[item.scenario];

      // 1. Measure Custom Engine Solo
      dom.benchStatusTitle.textContent = `計測中 (${step + 1}/${totalSteps}): 自作エンジン — ${sc.label}`;
      dom.benchStatusDetail.textContent = `${item.count.toLocaleString()} ${sc.entityUnit} · 単独フルスクリーン計測中...`;
      dom.benchProgressFill.style.width = `${Math.round((step / totalSteps) * 100)}%`;
      const customStats = await measureEngineSolo('custom', item.scenario, item.count, durationPerEngineMs);
      step++;
      if (state.benchmarkCancelled) break;

      // 2. Measure Three.js Solo
      dom.benchStatusTitle.textContent = `計測中 (${step + 1}/${totalSteps}): Three.js — ${sc.label}`;
      dom.benchStatusDetail.textContent = `${item.count.toLocaleString()} ${sc.entityUnit} · 単独フルスクリーン計測中...`;
      dom.benchProgressFill.style.width = `${Math.round((step / totalSteps) * 100)}%`;
      const threeStats = await measureEngineSolo('three', item.scenario, item.count, durationPerEngineMs);
      step++;
      if (state.benchmarkCancelled) break;

      results.push({
        scenario: item.scenario,
        label: sc.label,
        count: item.count,
        unit: sc.entityUnit,
        custom: customStats,
        three: threeStats,
      });
    }
    dom.benchProgressFill.style.width = '100%';
  } finally {
    dom.benchBanner.classList.add('hidden');
    state.benchmarkRunning = false;

    // Restore user's view & scenario cleanly
    updateViewModeUI(prevView);
    reconfigureEngines(prevScenario, prevCount);
  }

  if (!state.benchmarkCancelled && results.length > 0) {
    // Merge/replace by scenario id
    for (const r of results) {
      const idx = state.benchmarkResults.findIndex((x) => x.scenario === r.scenario && x.count === r.count);
      if (idx >= 0) state.benchmarkResults[idx] = r;
      else state.benchmarkResults.push(r);
    }
    renderReportModal();
    dom.reportModal.classList.remove('hidden');
  }

  return results;
}

function renderReportModal() {
  const list = state.benchmarkResults;
  const customStack = state.customEngine ? `${state.customEngine.backendLabel} + ${state.customEngine.wasmLabel}` : 'Custom';
  const threeStack = state.threeEngine ? state.threeEngine.backendLabel : 'Three.js';
  dom.reportEnvInfo.textContent = `${customStack}  vs  ${threeStack}  |  解像度 ${window.innerWidth}×${window.innerHeight} (DPR ${Math.min(window.devicePixelRatio || 1, 2)})`;

  if (list.length === 0) {
    dom.reportEmpty.classList.remove('hidden');
    dom.reportContent.classList.add('hidden');
    return;
  }

  dom.reportEmpty.classList.add('hidden');
  dom.reportContent.classList.remove('hidden');

  // Compute aggregate summary KPIs
  let cpuSpeedupProd = 1;
  let fpsRatioProd = 1;
  for (const r of list) {
    cpuSpeedupProd *= r.three.avgCpuMs / Math.max(0.01, r.custom.avgCpuMs);
    fpsRatioProd *= r.custom.fps / Math.max(0.1, r.three.fps);
  }
  const geoCpuSpeedup = Math.pow(cpuSpeedupProd, 1 / list.length);
  const geoFpsRatio = Math.pow(fpsRatioProd, 1 / list.length);

  const threeFootprint = state.threeBackend === 'webgpu'
    ? ENGINE_FOOTPRINT.threeWebGPU
    : ENGINE_FOOTPRINT.threeWebGL;
  const sizeRatio = threeFootprint.rawKB / ENGINE_FOOTPRINT.custom.rawKB;
  const initRatio = (state.threeEngine?.initMs || 1) / Math.max(1, state.customEngine?.initMs || 1);

  dom.reportKpis.innerHTML = `
    <div class="kpi-card">
      <div class="kpi-label">CPU処理速度 (幾何平均)</div>
      <div class="kpi-value ${geoCpuSpeedup < 1 ? 'accent-three' : ''}">${geoCpuSpeedup.toFixed(2)}x</div>
      <div class="kpi-sub">${geoCpuSpeedup >= 1 ? '自作エンジンが高速' : 'Three.jsが高速'}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">実効FPS比 (幾何平均)</div>
      <div class="kpi-value ${geoFpsRatio < 1 ? 'accent-three' : ''}">${geoFpsRatio.toFixed(2)}x</div>
      <div class="kpi-sub">${list.length}シナリオ単独計測</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">総バンドル軽量化</div>
      <div class="kpi-value">${sizeRatio.toFixed(1)}x</div>
      <div class="kpi-sub">${ENGINE_FOOTPRINT.custom.rawKB} KB vs ${threeFootprint.rawKB.toLocaleString()} KB</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">初期化・シェーダー構築</div>
      <div class="kpi-value ${initRatio < 1 ? 'accent-three' : ''}">${initRatio.toFixed(1)}x</div>
      <div class="kpi-sub">${(state.customEngine?.initMs || 0).toFixed(0)} ms vs ${(state.threeEngine?.initMs || 0).toFixed(0)} ms</div>
    </div>
  `;

  // Populate detailed table
  dom.reportTbody.innerHTML = '';
  for (const r of list) {
    const rows = [
      { cls: 'cell-custom', name: '自作エンジン', s: r.custom },
      { cls: 'cell-three', name: `Three.js (${state.threeBackend === 'webgpu' ? 'WebGPU' : 'WebGL2'})`, s: r.three },
    ];
    for (let i = 0; i < rows.length; i++) {
      const tr = document.createElement('tr');
      const { cls, name, s } = rows[i];
      tr.innerHTML = `
        <td>${i === 0 ? `<strong>${r.label}</strong> (${r.count.toLocaleString()}${r.unit})` : ''}</td>
        <td class="${cls}">${name}</td>
        <td><strong>${s.fps.toFixed(1)}</strong></td>
        <td>${s.avgFrameMs.toFixed(2)} ms</td>
        <td>${s.p99FrameMs.toFixed(2)} ms</td>
        <td><strong>${s.avgCpuMs.toFixed(2)} ms</strong></td>
        <td>${s.avgGpuMs >= 0 ? `${s.avgGpuMs.toFixed(2)} ms` : 'N/A'}</td>
        <td>${s.initMs.toFixed(0)} ms</td>
        <td>${s.heapMB >= 0 ? `${s.heapMB.toFixed(1)} MB` : 'N/A'}</td>
      `;
      dom.reportTbody.appendChild(tr);
    }
  }

  // Render crisp Canvas2D grouped bar charts on next frame so container width is laid out
  requestAnimationFrame(() => {
    drawBenchmarkBarChart(
      dom.chartCpu,
      list.map((r) => ({
        title: r.label,
        subtitle: `${r.count.toLocaleString()} ${r.unit} · CPU ms (低い方が高速)`,
        customVal: r.custom.avgCpuMs,
        threeVal: r.three.avgCpuMs,
        customText: `${r.custom.avgCpuMs.toFixed(2)} ms`,
        threeText: `${r.three.avgCpuMs.toFixed(2)} ms`,
      })),
    );
    drawBenchmarkBarChart(
      dom.chartFps,
      list.map((r) => ({
        title: r.label,
        subtitle: `${r.count.toLocaleString()} ${r.unit} · 平均FPS (高い方が高速)`,
        customVal: r.custom.fps,
        threeVal: r.three.fps,
        customText: `${r.custom.fps.toFixed(1)} FPS`,
        threeText: `${r.three.fps.toFixed(1)} FPS`,
      })),
    );
  });
}

// --- Event Wiring ---

function bindEvents() {
  state.camera.attachElement(dom.paneCustom, () => {
    dom.btnToggleOrbit.classList.toggle('active', state.camera.autoOrbit);
    dom.btnToggleOrbit.textContent = state.camera.autoOrbit ? '自動周回 ON' : '自動周回 OFF';
  });
  state.camera.attachElement(dom.paneThree, () => {
    dom.btnToggleOrbit.classList.toggle('active', state.camera.autoOrbit);
    dom.btnToggleOrbit.textContent = state.camera.autoOrbit ? '自動周回 ON' : '自動周回 OFF';
  });

  for (const btn of dom.scenarioBtns) {
    btn.addEventListener('click', () => {
      if (state.benchmarkRunning) return;
      const scId = btn.dataset.scenario;
      reconfigureEngines(scId, SCENARIOS[scId].defaultCount);
    });
  }

  for (const btn of dom.viewBtns) {
    btn.addEventListener('click', () => {
      if (state.benchmarkRunning) return;
      updateViewModeUI(btn.dataset.view);
    });
  }

  dom.selectCount.addEventListener('change', () => {
    if (state.benchmarkRunning) return;
    const count = parseInt(dom.selectCount.value, 10);
    reconfigureEngines(state.scenario, count);
  });

  dom.selectThreeBackend.addEventListener('change', async () => {
    if (state.benchmarkRunning) return;
    dom.threeStack.textContent = 'バックエンド切替中...';
    await initThreeEngine(dom.selectThreeBackend.value);
    handleResize();
  });

  dom.btnToggleOrbit.addEventListener('click', () => {
    state.camera.autoOrbit = !state.camera.autoOrbit;
    dom.btnToggleOrbit.classList.toggle('active', state.camera.autoOrbit);
    dom.btnToggleOrbit.textContent = state.camera.autoOrbit ? '自動周回 ON' : '自動周回 OFF';
  });

  dom.btnResetSim.addEventListener('click', () => {
    if (state.benchmarkRunning) return;
    state.camera.resetForBenchmark();
    dom.btnToggleOrbit.classList.add('active');
    dom.btnToggleOrbit.textContent = '自動周回 ON';
    reconfigureEngines(state.scenario, state.entityCount);
  });

  dom.btnRunBench.addEventListener('click', () => runAutomatedBenchmark(true, 2200));
  dom.btnBenchCurrent.addEventListener('click', () => runAutomatedBenchmark(false, 2200));
  dom.btnBenchAll.addEventListener('click', () => runAutomatedBenchmark(true, 2200));

  dom.btnCancelBench.addEventListener('click', () => {
    state.benchmarkCancelled = true;
  });

  dom.btnOpenReport.addEventListener('click', () => {
    renderReportModal();
    dom.reportModal.classList.remove('hidden');
  });

  dom.btnCloseReport.addEventListener('click', () => {
    dom.reportModal.classList.add('hidden');
  });

  dom.reportModal.addEventListener('click', (e) => {
    if (e.target === dom.reportModal) {
      dom.reportModal.classList.add('hidden');
    }
  });

  window.addEventListener('resize', handleResize);
}

// Expose deterministic programmatic hook for automated verification & reproducible benchmarking
window.__AETHER_BENCH__ = {
  state,
  runAutomatedBenchmark,
  reconfigureEngines,
  updateViewModeUI,
  initThreeEngine,
};

async function boot() {
  populateCountSelect(state.scenario, state.entityCount);
  bindEvents();
  syncTopbarOffset();
  const topbar = document.querySelector('.topbar');
  if (topbar && typeof ResizeObserver !== 'undefined') {
    new ResizeObserver(syncTopbarOffset).observe(topbar);
  }

  // Initialize Custom Engine first, then Three.js cleanly
  await initCustomEngine();
  await initThreeEngine(state.hasWebGPU ? 'webgpu' : 'webgl2');
  handleResize();

  requestAnimationFrame(animationLoop);
}

boot().catch((err) => {
  console.error('Boot error:', err);
  dom.customStack.textContent = `エラー: ${err.message}`;
});
