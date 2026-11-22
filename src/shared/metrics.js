// High-precision frame telemetry collector + zero-dependency Canvas2D sparkline & benchmark chart renderer

export class FrameTelemetry {
  constructor(windowSize = 120) {
    this.windowSize = windowSize;
    this.frameMs = new Float32Array(windowSize);
    this.cpuMs = new Float32Array(windowSize);
    this.gpuMs = new Float32Array(windowSize);
    this.cursor = 0;
    this.samples = 0;
    this.initMs = 0;
    this.visibleCount = 0;
    this.backendLabel = '';
    this.wasmLabel = '';
    this.hasGpuTimestamps = false;

    this.recording = false;
    this.recFrames = [];
    this.recCpu = [];
    this.recGpu = [];
  }

  reset() {
    this.cursor = 0;
    this.samples = 0;
    this.frameMs.fill(0);
    this.cpuMs.fill(0);
    this.gpuMs.fill(0);
    this.recFrames.length = 0;
    this.recCpu.length = 0;
    this.recGpu.length = 0;
  }

  startRecording() {
    this.recording = true;
    this.recFrames.length = 0;
    this.recCpu.length = 0;
    this.recGpu.length = 0;
  }

  stopRecording() {
    this.recording = false;
    return this.computeStats(this.recFrames, this.recCpu, this.recGpu);
  }

  record(frameDeltaMs, cpuWorkMs, gpuPassMs = -1, visibleCount = 0) {
    const idx = this.cursor;
    this.frameMs[idx] = frameDeltaMs;
    this.cpuMs[idx] = cpuWorkMs;
    this.gpuMs[idx] = gpuPassMs >= 0 ? gpuPassMs : -1;
    if (gpuPassMs >= 0) this.hasGpuTimestamps = true;
    this.visibleCount = visibleCount;

    this.cursor = (idx + 1) % this.windowSize;
    if (this.samples < this.windowSize) this.samples++;

    if (this.recording) {
      this.recFrames.push(frameDeltaMs);
      this.recCpu.push(cpuWorkMs);
      if (gpuPassMs >= 0) this.recGpu.push(gpuPassMs);
    }
  }

  getLiveSummary() {
    const n = this.samples;
    if (n === 0) {
      return {
        fps: 0,
        avgFrameMs: 0,
        p95FrameMs: 0,
        p99FrameMs: 0,
        avgCpuMs: 0,
        avgGpuMs: -1,
        heapMB: readHeapMB(),
        initMs: this.initMs,
        visibleCount: this.visibleCount,
      };
    }
    const frames = [];
    const cpus = [];
    const gpus = [];
    for (let i = 0; i < n; i++) {
      frames.push(this.frameMs[i]);
      cpus.push(this.cpuMs[i]);
      if (this.gpuMs[i] >= 0) gpus.push(this.gpuMs[i]);
    }
    return this.computeStats(frames, cpus, gpus);
  }

  computeStats(frames, cpus, gpus) {
    const n = frames.length;
    if (n === 0) {
      return {
        fps: 0, avgFrameMs: 0, p50FrameMs: 0, p95FrameMs: 0, p99FrameMs: 0,
        avgCpuMs: 0, avgGpuMs: -1, heapMB: readHeapMB(), initMs: this.initMs,
        visibleCount: this.visibleCount, sampleCount: 0,
      };
    }
    let sumF = 0, sumC = 0, sumG = 0;
    for (let i = 0; i < n; i++) {
      sumF += frames[i];
      sumC += cpus[i];
    }
    for (let i = 0; i < gpus.length; i++) sumG += gpus[i];

    const sorted = frames.slice().sort((a, b) => a - b);
    const avgFrameMs = sumF / n;
    const avgCpuMs = sumC / n;
    const avgGpuMs = gpus.length > 0 ? sumG / gpus.length : -1;

    return {
      fps: avgFrameMs > 0 ? 1000 / avgFrameMs : 0,
      avgFrameMs,
      p50FrameMs: sorted[Math.floor(n * 0.50)],
      p95FrameMs: sorted[Math.min(n - 1, Math.floor(n * 0.95))],
      p99FrameMs: sorted[Math.min(n - 1, Math.floor(n * 0.99))],
      avgCpuMs,
      avgGpuMs,
      heapMB: readHeapMB(),
      initMs: this.initMs,
      visibleCount: this.visibleCount,
      sampleCount: n,
    };
  }
}

export function readHeapMB() {
  const mem = performance.memory;
  if (mem && typeof mem.usedJSHeapSize === 'number') {
    return mem.usedJSHeapSize / (1024 * 1024);
  }
  return -1;
}

/**
 * Renders a crisp high-DPI frame-time sparkline on a <canvas> element.
 */
export function drawSparkline(canvas, telemetry, accentColor) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = canvas.clientWidth || 160;
  const h = canvas.clientHeight || 34;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  ctx.save();
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  const n = telemetry.samples;
  if (n < 2) {
    ctx.restore();
    return;
  }

  // Reference 16.67ms (60 FPS) line
  const maxMs = 36;
  const y60 = h - (16.67 / maxMs) * h;
  ctx.strokeStyle = 'rgba(30, 41, 59, 0.14)';
  ctx.setLineDash([3, 3]);
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, y60);
  ctx.lineTo(w, y60);
  ctx.stroke();
  ctx.setLineDash([]);

  // Frame time curve
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const idx = (telemetry.cursor - n + i + telemetry.windowSize) % telemetry.windowSize;
    const ms = Math.min(maxMs, telemetry.frameMs[idx]);
    const x = (i / (telemetry.windowSize - 1)) * w;
    const y = h - (ms / maxMs) * (h - 4) - 2;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.strokeStyle = accentColor;
  ctx.lineWidth = 1.6;
  ctx.stroke();

  // Subtle fill under curve
  const lastX = ((n - 1) / (telemetry.windowSize - 1)) * w;
  ctx.lineTo(lastX, h);
  ctx.lineTo(0, h);
  ctx.closePath();
  ctx.fillStyle = accentColor.replace('rgb(', 'rgba(').replace(')', ', 0.13)');
  ctx.fill();

  ctx.restore();
}

/**
 * Renders a crisp horizontal grouped comparison bar chart into a Canvas2D element.
 */
export function drawBenchmarkBarChart(canvas, items) {
  const ctx = canvas.getContext('2d');
  if (!ctx || !items || items.length === 0) return;

  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = canvas.clientWidth || 640;
  const rowH = 74;
  const padTop = 28;
  const padBottom = 14;
  const h = padTop + items.length * rowH + padBottom;

  canvas.style.height = `${h}px`;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);

  ctx.save();
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  // Legend
  ctx.font = '600 11px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
  ctx.fillStyle = '#0f766e';
  ctx.fillRect(12, 8, 12, 10);
  ctx.fillStyle = '#1e293b';
  ctx.fillText('自作エンジン (Wasm SIMD + WebGPU)', 30, 17);

  ctx.fillStyle = '#d95b43';
  ctx.fillRect(245, 8, 12, 10);
  ctx.fillStyle = '#1e293b';
  ctx.fillText('Three.js v0.187', 263, 17);

  const labelW = Math.min(185, w * 0.30);
  const barAreaW = w - labelW - 108;

  items.forEach((item, i) => {
    const y = padTop + i * rowH;

    // Row divider
    if (i > 0) {
      ctx.strokeStyle = 'rgba(15, 23, 42, 0.08)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(10, y - 6);
      ctx.lineTo(w - 10, y - 6);
      ctx.stroke();
    }

    // Scenario / metric title
    ctx.fillStyle = '#0f172a';
    ctx.font = '600 12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    ctx.fillText(item.title, 12, y + 18);
    ctx.fillStyle = '#64748b';
    ctx.font = '500 10.5px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillText(item.subtitle, 12, y + 36);

    const maxVal = Math.max(item.customVal, item.threeVal, 0.001);
    const wCustom = Math.max(4, (item.customVal / maxVal) * barAreaW);
    const wThree = Math.max(4, (item.threeVal / maxVal) * barAreaW);

    // Custom bar
    ctx.fillStyle = '#0f766e';
    roundRect(ctx, labelW, y + 6, wCustom, 18, 4);
    ctx.fill();

    // Three bar
    ctx.fillStyle = '#d95b43';
    roundRect(ctx, labelW, y + 30, wThree, 18, 4);
    ctx.fill();

    // Value labels
    ctx.font = '600 11.5px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillStyle = '#0f766e';
    ctx.fillText(item.customText, labelW + wCustom + 8, y + 19);
    ctx.fillStyle = '#c2410c';
    ctx.fillText(item.threeText, labelW + wThree + 8, y + 43);
  });

  ctx.restore();
}

function roundRect(ctx, x, y, w, h, r) {
  const rr = Math.min(r, w * 0.5, h * 0.5);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}
