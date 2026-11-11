#!/usr/bin/env node
// Run against a local preview: npm run benchmark:chamber -- http://127.0.0.1:1111
// The seeded scene drives real controls; timings are diagnostic, never CI gates.
const { chromium } = require("@playwright/test");
const assert = require("node:assert/strict");

async function main() {
  const browser = await chromium.launch({
    ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH
      ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH } : {}),
  });
  try {
    const results = [];
    for (const mode of ["replay", "photograph"]) {
      const errors = [];
      const page = await browser.newPage({ viewport: { width: 1440, height: 900 },
        deviceScaleFactor: 2, colorScheme: "dark", reducedMotion: "reduce" });
      page.on("pageerror", (error) => errors.push(error.message));
      await page.addInitScript(() => {
        let seed = 42;
        Math.random = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) + 0.5) / 4294967296;
        const stats = { frames: [], gaps: [], arcs: 0, fills: 0, images: 0, longTasks: [] };
        window.chamberBenchmark = stats;
        for (const [method, key] of [["arc", "arcs"], ["fill", "fills"], ["drawImage", "images"]]) {
          const original = CanvasRenderingContext2D.prototype[method];
          CanvasRenderingContext2D.prototype[method] = function (...args) {
            stats[key] += 1;
            return original.apply(this, args);
          };
        }
        let previous = 0;
        const raf = window.requestAnimationFrame.bind(window);
        window.requestAnimationFrame = (callback) => raf((time) => {
          if (callback.name !== "animate") return callback(time);
          if (previous) stats.gaps.push(time - previous);
          previous = time;
          const start = performance.now();
          callback(time);
          stats.frames.push(performance.now() - start);
        });
        new PerformanceObserver((list) => {
          stats.longTasks.push(...list.getEntries().map((entry) => entry.duration));
        }).observe({ type: "longtask", buffered: false });
      });
      page.setDefaultTimeout(15000);
      await page.goto(`${process.argv[2] || "http://127.0.0.1:1111"}/playground/`);
      await page.evaluate(() => {
        const stats = window.chamberBenchmark;
        stats.layerTracks = 0; stats.preparedSlices = 0; stats.finishedLayers = 0; stats.layers = 0;
        const create = BubbleRenderer.createLayer;
        BubbleRenderer.createLayer = (...args) => {
          stats.layers += 1;
          const layer = create(...args), add = layer.addTrack, prepare = layer.prepare;
          layer.addTrack = (track) => { stats.layerTracks += 1; return add(track); };
          layer.prepare = function* () {
            for (const step of prepare()) { stats.preparedSlices += 1; yield step; }
            stats.finishedLayers += 1;
          };
          return layer;
        };
      });
      await page.locator(".playground-beam > summary").click();
      await page.locator("#playground-beam-density").fill("60");
      await page.locator("#playground-event-rate").fill("10");
      await page.selectOption("#playground-view", mode);
      await page.getByRole("button", { name: "Resume", exact: true }).click();
      await page.evaluate(() => {
        const stats = window.chamberBenchmark;
        for (const key of ["frames", "gaps", "longTasks"]) stats[key].length = 0;
        stats.arcs = stats.fills = stats.images = 0;
      });
      await page.getByRole("button", { name: mode === "photograph" ? "Next photo" : "Clear", exact: true }).click();
      await page.waitForTimeout(6500);
      const stats = await page.evaluate(() => window.chamberBenchmark);
      const pending = await page.locator(".playground-beam").getAttribute("aria-busy");
      assert.equal(pending, null, `${mode} beam preparation must finish; ${JSON.stringify({
        preparedTracks: stats.layerTracks, slices: stats.preparedSlices,
        finishedLayers: stats.finishedLayers, layers: stats.layers, errors,
      })}`);
      assert.deepEqual(errors, [], "benchmark must have no browser exceptions");
      assert.ok(stats.frames.length >= 30, "benchmark must exercise ongoing rendering");
      const percentile = (array, p) => {
        const sorted = [...array].sort((a, b) => a - b);
        return Number((sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))] || 0).toFixed(2));
      };
      results.push({ mode, beamTracks: 60, requestedMeanReactionsPerBeam: 10, trackBudget: 96, frames: stats.frames.length,
        preparedTracks: stats.layerTracks, preparedSlices: stats.preparedSlices,
        frameCpuMedian: percentile(stats.frames, 0.5), frameCpuP95: percentile(stats.frames, 0.95),
        frameGapP95: percentile(stats.gaps, 0.95), longTasks: stats.longTasks.length,
        longestTask: percentile(stats.longTasks, 1), arcCalls: stats.arcs, fillCalls: stats.fills,
        imageCalls: stats.images });
      await page.close();
    }
    console.log(JSON.stringify(results, null, 2));
    if (process.env.CHAMBER_PERF_ASSERT === "1") {
      // Use opt-in budgets only on a controlled machine. Structural regression
      // tests separately enforce bounded per-frame drawing work in ordinary CI.
      for (const result of results) assert.ok(result.frameCpuP95 < 16.7,
        `${result.mode} p95 frame CPU ${result.frameCpuP95} ms exceeds a 60 Hz frame budget`);
    }
  } finally {
    await browser.close();
  }
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
