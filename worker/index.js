// Zero-dependency Cloudflare Worker for automatic benchmark result collection (backed by Cloudflare D1)

const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
  'Access-Control-Max-Age': '86400',
};

const VALID_SCENARIOS = new Set(['hybrid', 'gpu-compute', 'scene-graph']);

function jsonResponse(body, status = 200, extraHeaders = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      'Content-Type': 'application/json; charset=utf-8',
      ...CORS_HEADERS,
      ...extraHeaders,
    },
  });
}

function clampStr(val, maxLen = 512) {
  if (val === null || val === undefined) return null;
  const s = String(val);
  return s.length > maxLen ? s.slice(0, maxLen) : s;
}

function finiteNum(val, fallback = null) {
  const n = Number(val);
  return Number.isFinite(n) ? n : fallback;
}

function toIntFlag(val) {
  if (val === null || val === undefined) return null;
  return val ? 1 : 0;
}

export default {
  async fetch(request, env) {
    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: CORS_HEADERS });
    }

    const url = new URL(request.url);
    const path = url.pathname.replace(/\/+$/, '') || '/';

    try {
      if (path === '/' && request.method === 'GET') {
        return jsonResponse({
          service: 'AETHER FLOCK Benchmark Telemetry API',
          endpoints: {
            health: 'GET /api/health',
            submit: 'POST /api/results',
            list: 'GET /api/results?limit=30&scenario=hybrid',
            detail: 'GET /api/results/:id',
            summary: 'GET /api/summary',
          },
        });
      }

      if (path === '/api/health' && request.method === 'GET') {
        const row = await env.DB.prepare('SELECT COUNT(*) AS totalRuns FROM benchmark_runs').first();
        return jsonResponse({
          ok: true,
          service: 'threejs-vs-no-threejs-telemetry',
          totalRuns: row?.totalRuns ?? 0,
        });
      }

      if (path === '/api/results' && request.method === 'POST') {
        return await handleSubmitResult(request, env);
      }

      if (path === '/api/results' && request.method === 'GET') {
        return await handleListResults(url, env);
      }

      if (path.startsWith('/api/results/') && request.method === 'GET') {
        const id = path.slice('/api/results/'.length);
        return await handleGetResultById(id, env);
      }

      if (path === '/api/summary' && request.method === 'GET') {
        return await handleSummary(env);
      }

      return jsonResponse({ ok: false, error: 'Not Found' }, 404);
    } catch (err) {
      return jsonResponse({ ok: false, error: 'Internal Server Error', detail: String(err?.message || err) }, 500);
    }
  },
};

async function handleSubmitResult(request, env) {
  const rawText = await request.text();
  if (!rawText || rawText.length > 65536) {
    return jsonResponse({ ok: false, error: 'Payload empty or exceeds 64KB limit' }, 400);
  }

  let payload;
  try {
    payload = JSON.parse(rawText);
  } catch {
    return jsonResponse({ ok: false, error: 'Invalid JSON body' }, 400);
  }

  if (!payload || typeof payload !== 'object' || !Array.isArray(payload.results) || payload.results.length === 0) {
    return jsonResponse({ ok: false, error: 'Expected { device, results: [...] }' }, 400);
  }
  if (payload.results.length > 10) {
    return jsonResponse({ ok: false, error: 'Too many scenario results in a single submission' }, 400);
  }

  const device = (payload.device && typeof payload.device === 'object') ? payload.device : {};
  const runId = crypto.randomUUID();
  const createdAt = new Date().toISOString();

  const headerUA = request.headers.get('user-agent') || '';
  const userAgent = clampStr(device.userAgent || headerUA, 1024) || 'unknown';
  const uaData = device.userAgentData || {};
  const scr = device.screen || {};
  const vp = device.viewport || {};
  const fs = device.fullscreen || {};
  const wasm = device.wasm || {};
  const wgpu = device.webgpu || {};
  const gpuInfo = wgpu.adapterInfo || {};
  const wgl = device.webgl || {};
  const eng = device.engines || {};
  const cf = request.cf || {};

  // Enrich stored device JSON with non-PII Cloudflare edge location & header UA
  const enrichedDevice = {
    ...device,
    userAgent,
    serverUserAgent: clampStr(headerUA, 1024),
    cf: {
      country: cf.country || null,
      colo: cf.colo || null,
    },
  };

  const validResults = [];
  for (const item of payload.results) {
    if (!item || !VALID_SCENARIOS.has(item.scenario)) continue;
    const count = finiteNum(item.count, 0);
    if (count <= 0 || count > 262144) continue;
    const c = item.custom || {};
    const t = item.three || {};
    const customCpu = finiteNum(c.avgCpuMs, 0);
    const threeCpu = finiteNum(t.avgCpuMs, 0);
    const customFps = finiteNum(c.fps, 0);
    const threeFps = finiteNum(t.fps, 0);

    validResults.push({
      scenario: item.scenario,
      count: Math.round(count),
      custom: {
        fps: customFps,
        avgFrameMs: finiteNum(c.avgFrameMs, 0),
        p95FrameMs: finiteNum(c.p95FrameMs, 0),
        p99FrameMs: finiteNum(c.p99FrameMs, 0),
        avgCpuMs: customCpu,
        avgGpuMs: finiteNum(c.avgGpuMs, -1),
        heapMB: finiteNum(c.heapMB, -1),
        sampleCount: Math.round(finiteNum(c.sampleCount, 0)),
      },
      three: {
        fps: threeFps,
        avgFrameMs: finiteNum(t.avgFrameMs, 0),
        p95FrameMs: finiteNum(t.p95FrameMs, 0),
        p99FrameMs: finiteNum(t.p99FrameMs, 0),
        avgCpuMs: threeCpu,
        avgGpuMs: finiteNum(t.avgGpuMs, -1),
        heapMB: finiteNum(t.heapMB, -1),
        sampleCount: Math.round(finiteNum(t.sampleCount, 0)),
      },
      cpuSpeedup: customCpu > 0.001 ? threeCpu / customCpu : null,
      fpsRatio: threeFps > 0.001 ? customFps / threeFps : null,
    });
  }

  if (validResults.length === 0) {
    return jsonResponse({ ok: false, error: 'No valid benchmark scenario records provided' }, 400);
  }

  const statements = [];

  statements.push(
    env.DB.prepare(`
      INSERT INTO benchmark_runs (
        id, created_at, app_version, page_url,
        user_agent, ua_platform, ua_mobile, ua_model, ua_arch, ua_bitness,
        platform, language, cpu_cores, device_memory_gb, max_touch_points,
        screen_width, screen_height, viewport_width, viewport_height, dpr,
        orientation_type, fullscreen, wasm_simd, webgpu_supported,
        gpu_vendor, gpu_arch, gpu_device, gpu_description, gpu_is_fallback,
        webgl_renderer, webgl_vendor, custom_backend, three_backend,
        custom_init_ms, three_init_ms, cf_country, cf_colo,
        device_json, results_json
      ) VALUES (
        ?, ?, ?, ?,
        ?, ?, ?, ?, ?, ?,
        ?, ?, ?, ?, ?,
        ?, ?, ?, ?, ?,
        ?, ?, ?, ?,
        ?, ?, ?, ?, ?,
        ?, ?, ?, ?,
        ?, ?, ?, ?,
        ?, ?
      )
    `).bind(
      runId,
      createdAt,
      clampStr(payload.appVersion || '1.0.0', 32),
      clampStr(device.pageUrl, 512),
      userAgent,
      clampStr(uaData.platform, 64),
      toIntFlag(uaData.mobile),
      clampStr(uaData.model, 128),
      clampStr(uaData.architecture, 64),
      clampStr(uaData.bitness, 16),
      clampStr(device.platform, 128),
      clampStr(device.language, 32),
      finiteNum(device.hardwareConcurrency, null),
      finiteNum(device.deviceMemoryGB, null),
      finiteNum(device.maxTouchPoints, null),
      finiteNum(scr.width, null),
      finiteNum(scr.height, null),
      finiteNum(vp.width, null),
      finiteNum(vp.height, null),
      finiteNum(vp.dpr, null),
      clampStr(scr.orientationType, 64),
      toIntFlag(fs.active),
      toIntFlag(wasm.simd),
      toIntFlag(wgpu.supported),
      clampStr(gpuInfo.vendor, 128),
      clampStr(gpuInfo.architecture, 128),
      clampStr(gpuInfo.device, 128),
      clampStr(gpuInfo.description, 256),
      toIntFlag(gpuInfo.isFallbackAdapter),
      clampStr(wgl.unmaskedRenderer || wgl.renderer, 256),
      clampStr(wgl.unmaskedVendor || wgl.vendor, 128),
      clampStr(eng.customBackend, 128),
      clampStr(eng.threeBackend, 128),
      finiteNum(eng.customInitMs, null),
      finiteNum(eng.threeInitMs, null),
      clampStr(cf.country, 16),
      clampStr(cf.colo, 16),
      JSON.stringify(enrichedDevice),
      JSON.stringify(validResults),
    ),
  );

  const insertScenario = env.DB.prepare(`
    INSERT INTO benchmark_scenarios (
      run_id, created_at, scenario, entity_count,
      custom_fps, custom_frame_ms, custom_p95_ms, custom_p99_ms, custom_cpu_ms, custom_gpu_ms, custom_heap_mb, custom_samples,
      three_fps, three_frame_ms, three_p95_ms, three_p99_ms, three_cpu_ms, three_gpu_ms, three_heap_mb, three_samples,
      cpu_speedup, fps_ratio
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);

  for (const r of validResults) {
    statements.push(
      insertScenario.bind(
        runId,
        createdAt,
        r.scenario,
        r.count,
        r.custom.fps,
        r.custom.avgFrameMs,
        r.custom.p95FrameMs,
        r.custom.p99FrameMs,
        r.custom.avgCpuMs,
        r.custom.avgGpuMs,
        r.custom.heapMB,
        r.custom.sampleCount,
        r.three.fps,
        r.three.avgFrameMs,
        r.three.p95FrameMs,
        r.three.p99FrameMs,
        r.three.avgCpuMs,
        r.three.avgGpuMs,
        r.three.heapMB,
        r.three.sampleCount,
        r.cpuSpeedup,
        r.fpsRatio,
      ),
    );
  }

  await env.DB.batch(statements);

  return jsonResponse({
    ok: true,
    id: runId,
    created_at: createdAt,
    savedScenarios: validResults.length,
  }, 201);
}

async function handleListResults(url, env) {
  const limit = Math.max(1, Math.min(100, parseInt(url.searchParams.get('limit') || '25', 10) || 25));
  const scenarioFilter = url.searchParams.get('scenario');

  const countRow = await env.DB.prepare('SELECT COUNT(*) AS total FROM benchmark_runs').first();

  let rows;
  if (scenarioFilter && VALID_SCENARIOS.has(scenarioFilter)) {
    const res = await env.DB.prepare(`
      SELECT r.id, r.created_at, r.user_agent, r.ua_platform, r.ua_mobile, r.ua_model,
             r.cpu_cores, r.device_memory_gb, r.viewport_width, r.viewport_height, r.dpr,
             r.orientation_type, r.fullscreen, r.wasm_simd, r.webgpu_supported,
             r.gpu_vendor, r.gpu_arch, r.gpu_description, r.webgl_renderer,
             r.custom_backend, r.three_backend, r.custom_init_ms, r.three_init_ms,
             r.cf_country, r.results_json
      FROM benchmark_runs r
      WHERE EXISTS (SELECT 1 FROM benchmark_scenarios s WHERE s.run_id = r.id AND s.scenario = ?)
      ORDER BY r.created_at DESC
      LIMIT ?
    `).bind(scenarioFilter, limit).all();
    rows = res.results || [];
  } else {
    const res = await env.DB.prepare(`
      SELECT id, created_at, user_agent, ua_platform, ua_mobile, ua_model,
             cpu_cores, device_memory_gb, viewport_width, viewport_height, dpr,
             orientation_type, fullscreen, wasm_simd, webgpu_supported,
             gpu_vendor, gpu_arch, gpu_description, webgl_renderer,
             custom_backend, three_backend, custom_init_ms, three_init_ms,
             cf_country, results_json
      FROM benchmark_runs
      ORDER BY created_at DESC
      LIMIT ?
    `).bind(limit).all();
    rows = res.results || [];
  }

  const runs = rows.map((row) => {
    let results = [];
    try { results = JSON.parse(row.results_json); } catch {}
    return {
      id: row.id,
      createdAt: row.created_at,
      userAgent: row.user_agent,
      uaPlatform: row.ua_platform,
      uaMobile: row.ua_mobile === 1,
      uaModel: row.ua_model,
      cpuCores: row.cpu_cores,
      deviceMemoryGB: row.device_memory_gb,
      viewportWidth: row.viewport_width,
      viewportHeight: row.viewport_height,
      dpr: row.dpr,
      orientationType: row.orientation_type,
      fullscreen: row.fullscreen === 1,
      wasmSimd: row.wasm_simd === 1,
      webgpuSupported: row.webgpu_supported === 1,
      gpuVendor: row.gpu_vendor,
      gpuArch: row.gpu_arch,
      gpuDescription: row.gpu_description,
      webglRenderer: row.webgl_renderer,
      customBackend: row.custom_backend,
      threeBackend: row.three_backend,
      customInitMs: row.custom_init_ms,
      threeInitMs: row.three_init_ms,
      country: row.cf_country,
      results,
    };
  });

  return jsonResponse({
    ok: true,
    totalRuns: countRow?.total ?? 0,
    runs,
  });
}

async function handleGetResultById(id, env) {
  if (!id || id.length > 64) {
    return jsonResponse({ ok: false, error: 'Invalid ID' }, 400);
  }
  const row = await env.DB.prepare('SELECT * FROM benchmark_runs WHERE id = ?').bind(id).first();
  if (!row) {
    return jsonResponse({ ok: false, error: 'Not Found' }, 404);
  }
  let device = null;
  let results = [];
  try { device = JSON.parse(row.device_json); } catch {}
  try { results = JSON.parse(row.results_json); } catch {}

  return jsonResponse({
    ok: true,
    id: row.id,
    createdAt: row.created_at,
    appVersion: row.app_version,
    device,
    results,
  });
}

async function handleSummary(env) {
  const [totalRow, scenarioRes] = await Promise.all([
    env.DB.prepare('SELECT COUNT(*) AS totalRuns, COUNT(DISTINCT COALESCE(webgl_renderer, gpu_description, gpu_arch)) AS uniqueGpus FROM benchmark_runs').first(),
    env.DB.prepare(`
      SELECT
        scenario,
        COUNT(*) AS samples,
        ROUND(AVG(custom_fps), 2) AS avgCustomFps,
        ROUND(AVG(three_fps), 2) AS avgThreeFps,
        ROUND(AVG(custom_cpu_ms), 3) AS avgCustomCpuMs,
        ROUND(AVG(three_cpu_ms), 3) AS avgThreeCpuMs,
        ROUND(AVG(cpu_speedup), 2) AS avgCpuSpeedup,
        ROUND(AVG(fps_ratio), 2) AS avgFpsRatio
      FROM benchmark_scenarios
      GROUP BY scenario
      ORDER BY scenario
    `).all(),
  ]);

  return jsonResponse({
    ok: true,
    totalRuns: totalRow?.totalRuns ?? 0,
    uniqueGpus: totalRow?.uniqueGpus ?? 0,
    scenarios: scenarioRes.results || [],
  });
}
