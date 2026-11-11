const { test, expect } = require("@playwright/test");
const AxeBuilder = require("@axe-core/playwright").default;

const pages = [
  ["/", "Dr. Dene Hoffman"],
  ["/projects/", "Projects"],
  ["/projects/laddu/", "laddu"],
  ["/projects/maryada/", "maryada"],
  ["/projects/yamloom/", "yamloom"],
  ["/projects/pdg-rs/", "pdg-rs"],
  ["/projects/splotrs/", "splotrs"],
  ["/projects/see/", "see"],
  ["/projects/hddm-rs/", "hddm-rs"],
  ["/publications/", "Publications"],
  ["/research/", "Research"],
  ["/blog/", "String Theory"],
  ["/projects/gluex-rs/", "gluex-rs"],
  ["/blog/the-bfgs-algorithm-family-in-rust-part-3/", "The BFGS Algorithm Family in Rust (Part 3)"],
];

for (const [path, heading] of pages) {
  test(`${path} renders without serious accessibility or overflow problems`, async ({ page }) => {
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto(path);
    await expect(page.getByRole("heading", { level: 1, name: heading }).filter({ visible: true })).toBeVisible();
    const dimensions = await page.evaluate(() => {
      const clientWidth = document.documentElement.clientWidth;
      const overflowing = [...document.querySelectorAll("body *")]
        .map((element) => {
          const rect = element.getBoundingClientRect();
          return {
            selector: `${element.tagName.toLowerCase()}${element.id ? `#${element.id}` : ""}${[...element.classList].map((name) => `.${name}`).join("")}`,
            left: Math.round(rect.left),
            right: Math.round(rect.right),
            width: Math.round(rect.width),
          };
        })
        .filter(({ left, right }) => left < -1 || right > clientWidth + 1)
        .slice(0, 12);
      return {
        clientWidth,
        scrollWidth: document.documentElement.scrollWidth,
        overflowing,
      };
    });
    expect(dimensions, `Overflowing elements: ${JSON.stringify(dimensions.overflowing)}`).toMatchObject({
      clientWidth: dimensions.scrollWidth,
    });

    const results = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
      .exclude(".giallo")
      .analyze();
    const serious = results.violations.filter(({ impact }) => impact === "serious" || impact === "critical");
    expect(errors).toEqual([]);
    expect(serious.map(({ id, nodes }) => ({
      id,
      targets: nodes.map((node) => node.target.join(" ")),
    }))).toEqual([]);
  });
}

test("project status badges are reserved for exceptional states", async ({ page }) => {
  await page.goto("/projects/");
  await expect(page.locator(".project-directory__card")).toHaveCount(9);
  await expect(page.locator(".project-directory__status", { hasText: /^Archived$/ })).toHaveCount(0);
  await expect(page.locator(".project-directory__status", { hasText: /^Experimental$/ })).toHaveCount(1);
  await expect(page.locator(".project-directory__status", { hasText: /^Active/ })).toHaveCount(0);
});

test("publication topics filter records and report the result", async ({ page }) => {
  await page.goto("/publications/");
  await page.getByRole("button", { name: "Materials" }).click();
  await expect(page.locator(".publication-filters__status")).toHaveText("2 publications");
  await expect(page.getByRole("button", { name: "Materials" })).toHaveAttribute("aria-pressed", "true");
});

test("custom scripts are loaded only on relevant pages", async ({ page }) => {
  const scriptNames = () => page.locator("script[src]").evaluateAll((scripts) =>
    scripts.map((script) => new URL(script.src).pathname.split("/").pop()),
  );

  await page.goto("/");
  expect(await scriptNames()).toContain("home-links.js");
  expect(await scriptNames()).not.toContain("publication-records.js");

  await page.goto("/publications/");
  expect(await scriptNames()).toContain("publication-records.js");
  expect(await scriptNames()).not.toContain("laddu-interference.js");

  await page.goto("/projects/laddu/");
  expect(await scriptNames()).toContain("laddu-interference.js");
  expect(await scriptNames()).not.toContain("publication-records.js");

  await page.goto("/projects/maryada/");
  expect(await scriptNames()).toContain("maryada-branch-and-bound.js");
  expect(await scriptNames()).not.toContain("laddu-interference.js");
});

test("Maryada branch-and-bound explorer exposes inspectable bounds", async ({ page }) => {
  await page.goto("/projects/maryada/");
  const demo = page.locator("[data-maryada-bnb]");

  await expect(demo.getByRole("button", { name: /Take next step/ })).toBeEnabled();
  await expect(demo.locator("tbody [data-maryada-branch-select]")).toHaveCount(1);

  await demo.getByRole("button", { name: /Take next step/ }).click();
  await expect(demo.locator("tbody [data-maryada-branch-select]")).toHaveCount(3);
  await expect(demo.locator("[data-maryada-branches]")).toContainText("pruned");

  await demo.locator('tbody [data-maryada-branch-select="branch-2"]').click();
  await expect(demo.locator(".maryada-bnb__inspector h3")).toContainText("B2");
  await expect(demo.locator(".maryada-bnb__inspector code")).toContainText("f(X)");
});

async function quietChamber(page) {
  await page.locator("#playground-event-rate").fill("0");
  await page.selectOption("#playground-topology", "dalitz");
  await page.locator(".playground-beam > summary").click();
  await page.locator("#playground-beam-density").fill("0");
  await page.locator(".playground-beam > summary").click();
  await page.getByRole("button", { name: "Clear", exact: true }).click();
}

async function chamberInk(page) {
  return page.locator("canvas.bubble-chamber").evaluate((canvas) => {
    const pixels = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
    let ink = 0;
    for (let index = 3; index < pixels.length; index += 4) ink += pixels[index];
    return ink;
  });
}

test("playground is linked from the chamber controls, outside the navigation", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#site-nav a[href='/playground/']")).toHaveCount(0);
  await page.getByLabel("About the bubble chamber", { exact: true }).click();
  await expect(page.locator(".bubble-chamber-info").getByRole("link", { name: "Open particle playground" })).toBeVisible();
  await page.locator(".bubble-chamber-info").getByRole("link", { name: "Open particle playground" }).click();
  await expect(page).toHaveURL(/\/playground\/$/);
});


test("site search loads its index and finds projects", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  const navigation = page.getByRole("button", { name: "Open navigation", exact: true });
  if (await navigation.isVisible()) await navigation.click();
  const indexLoaded = page.waitForResponse((response) => response.url().includes("search_index.en.json"));
  await page.locator("#search-toggle").click();
  expect((await indexLoaded).ok()).toBe(true);
  await page.locator("#search-bar").fill("laddu");
  await expect(page.locator("#search-results .result-title").first()).toBeVisible();
  await expect(page.locator("#search-results")).toContainText("laddu");
  expect(errors).toEqual([]);
});


test("publications include the BESIII records from Scholar with publication status", async ({ page }) => {
  await page.goto("/publications/");
  const besiii = page.locator('[data-collaboration="besiii"]');
  await expect(besiii).toHaveCount(11);
  await expect(besiii.first()).toContainText("Published");
  await expect(besiii.first().getByRole("link")).toHaveAttribute("href", "https://doi.org/10.1103/ml3k-2pss");
  await expect(besiii.filter({ hasText: "Preprint" })).toHaveCount(10);
  const gluex = page.locator('[data-collaboration="gluex"]');
  await expect(gluex).toHaveCount(12);
  await expect(gluex.locator(":scope > span")).toHaveText(Array(12).fill("Published"));
  await page.getByRole("button", { name: "Materials", exact: true }).click();
  await expect(page.locator(".collaboration-publications")).toBeHidden();
});


// Drive only the controls visitors see. The background specks deliberately
// remain in these scenes; Clear removes particle tracks, not the chamber.
async function seedChamber(page, initial = 42) {
  await page.addInitScript((value) => {
    let seed = value;
    Math.random = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) + 0.5) / 4294967296;
  }, initial);
}

async function finishBeam(page) {
  // Preparation now yields through native message tasks, not simulation timers.
  // Advancing the fake playback clock does not itself await those tasks.
  await expect(page.locator(".playground-beam")).not.toHaveAttribute("aria-busy", "true");
}

async function chamberImage(page) {
  return page.locator("canvas.bubble-chamber").evaluate((canvas) => canvas.toDataURL());
}

async function tapChamber(page) {
  const viewport = page.viewportSize();
  await page.mouse.click(viewport.width * 0.75, viewport.height * 0.8);
}


test("compact controls retain reactions and pass accessibility checks", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/playground/");
  await expect(page.locator("#playground-topology option")).toHaveCount(5);
  await expect(page.locator(".playground-panel input[type=range]:visible")).toHaveCount(2);
  await expect(page.locator(".playground-panel input[type=checkbox]")).toHaveCount(1);
  await page.locator(".playground-beam > summary").click();
  await expect(page.locator("#playground-beam-density")).toBeVisible();
  await expect(page.locator("#playground-beam-direction")).toBeVisible();
  await page.selectOption("#playground-view", "photograph");
  await expect(page.locator("#playground-motion")).toBeHidden();
  await expect(page.getByRole("button", { name: "Next photo", exact: true })).toBeVisible();
  const results = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]).analyze();
  expect(results.violations.filter(({ impact }) => ["serious", "critical"].includes(impact)).map(({ id }) => id)).toEqual([]);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  expect(errors).toEqual([]);
});

test("playground pauses, saves a PNG, and clears particles", async ({ page }) => {
  await seedChamber(page);
  await page.clock.install();
  await page.goto("/playground/");
  await quietChamber(page);
  const empty = await chamberImage(page);
  await page.locator(".playground-panel > summary").click();
  await tapChamber(page);
  await page.clock.runFor(1500);
  expect(await chamberImage(page)).not.toBe(empty);
  await expect(page.locator(".bubble-chamber-status")).not.toContainText("generated");
  await page.locator(".playground-panel > summary").click();
  await page.getByRole("button", { name: "Pause", exact: true }).click();
  const frozen = await chamberImage(page);
  await page.clock.runFor(30000);
  expect(await chamberImage(page)).toBe(frozen);
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "Save PNG", exact: true }).click();
  const download = await downloadPromise;
  const png = require("node:fs").readFileSync(await download.path());
  expect([...png.subarray(0, 8)]).toEqual([137, 80, 78, 71, 13, 10, 26, 10]);
  expect(png.readUInt32BE(16)).toBeGreaterThanOrEqual(page.viewportSize().width);
  await page.getByRole("button", { name: "Clear", exact: true }).click();
  expect(await chamberImage(page)).toBe(empty);
});

test("trails expire automatically and reduced-motion starts paused", async ({ page }) => {
  await seedChamber(page);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.clock.install();
  await page.goto("/playground/");
  await quietChamber(page);
  await expect(page.getByRole("button", { name: "Resume", exact: true })).toBeVisible();
  const empty = await chamberImage(page);
  await page.getByRole("button", { name: "Resume", exact: true }).click();
  await page.locator("#playground-speed").fill("2");
  await page.locator(".playground-panel > summary").click();
  await tapChamber(page);
  await page.clock.runFor(500);
  expect(await chamberImage(page)).not.toBe(empty);
  // Isolate the lifetime check from incidental cosmic arrivals.
  await page.evaluate(() => { Math.random = () => 0.999999; });
  await page.clock.runFor(20000);
  expect(await chamberImage(page)).toBe(empty);
});

test("photographs fade in, stay fixed, then reset without fading out", async ({ page }) => {
  await seedChamber(page);
  await page.clock.install();
  await page.goto("/playground/");
  await quietChamber(page);
  await page.selectOption("#playground-view", "photograph");
  await page.locator(".playground-panel > summary").click();
  await tapChamber(page);
  await page.clock.runFor(200);
  const growing = await chamberInk(page);
  await page.clock.runFor(1000);
  expect(await chamberInk(page)).toBeGreaterThan(growing);
  const developed = await chamberImage(page);
  await page.clock.runFor(6300);
  expect(await chamberImage(page)).toBe(developed);
  await page.clock.runFor(1000);
  expect(await chamberImage(page)).not.toBe(developed);
  await page.locator(".playground-panel > summary").click();
  await page.getByRole("button", { name: "Pause", exact: true }).click();
  await page.getByRole("button", { name: "Next photo", exact: true }).click();
  const still = await chamberImage(page);
  await page.clock.runFor(30000);
  expect(await chamberImage(page)).toBe(still);
});

// Inspect real transport output at the renderer boundary; no alternate simulator.
async function observeChamber(page) {
  await page.evaluate(() => {
    window.chamberProbe = { transports: [], active: 0, maxActive: 0, beams: 0, maxBeams: 0, hold: 0 };
    const probe = window.chamberProbe;
    const create = BubbleRenderer.createLayer;
    BubbleRenderer.createLayer = (...args) => {
      const layer = create(...args);
      let count = 0, beam = false, shown = false;
      const add = layer.addTrack, draw = layer.draw, dispose = layer.dispose;
      layer.addTrack = (track) => {
        probe.transports.push({ mass: track.particle.mass, role: track.role, stopReason: track.stopReason, interaction: track.interaction, endState: track.states.at(-1), times: track.times, delay: track.delay, start: track.points[0], points: track.points });
        count += 1;
        if (track.role === "beam-incoming") beam = true;
        return add(track);
      };
      layer.draw = (...params) => {
        probe.hold = params[1].hold;
        if (!shown) {
          shown = true;
          probe.active += count;
          if (beam) probe.beams += 1;
          probe.maxActive = Math.max(probe.maxActive, probe.active);
          probe.maxBeams = Math.max(probe.maxBeams, probe.beams);
        }
        return draw(...params);
      };
      layer.dispose = () => {
        if (shown) { probe.active -= count; if (beam) probe.beams -= 1; shown = false; }
        return dispose();
      };
      return layer;
    };
  });
}

test("automatic reactions start on the beam and beam passages never overlap", async ({ page }) => {
  await seedChamber(page, 52);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.clock.install();
  await page.goto("/playground/");
  await observeChamber(page);
  await page.selectOption("#playground-topology", "dalitz");
  await page.locator("#playground-event-rate").fill("10");
  await page.locator(".playground-beam > summary").click();
  await page.locator("#playground-beam-density").fill("1");
  await page.locator("#playground-beam-direction").fill("90");
  await page.getByRole("button", { name: "Resume", exact: true }).click();
  await page.clock.runFor(5000);
  await finishBeam(page);
  const probe = await page.evaluate(() => window.chamberProbe);
  const beam = probe.transports.find((track) => track.role === "beam-incoming");
  const electrons = probe.transports.filter((track) => track.mass < 0.001);
  expect(beam).toBeTruthy();
  expect(Math.abs(beam.points.at(-1).y - beam.points[0].y)).toBeGreaterThan(3 * Math.abs(beam.points.at(-1).x - beam.points[0].x));
  expect(electrons.length).toBeGreaterThan(0);
  for (const track of electrons) {
    expect(Math.min(...beam.points.map((point) => Math.hypot(point.x - track.start.x, point.y - track.start.y)))).toBeLessThan(0.001);
  }
  await page.clock.runFor(25000);
  await finishBeam(page);
  const after = await page.evaluate(() => window.chamberProbe);
  expect(after.transports.filter((track) => track.role === "beam-incoming").length).toBeGreaterThan(1);
  expect(after.maxBeams).toBe(1);
  expect(after.maxActive).toBeLessThanOrEqual(96);
});

test("turning off the beam stops automatic reactions but allows manual placement", async ({ page }) => {
  await seedChamber(page);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.clock.install();
  await page.goto("/playground/");
  await observeChamber(page);
  await page.selectOption("#playground-topology", "dalitz");
  await page.locator(".playground-beam > summary").click();
  await page.locator("#playground-beam-density").fill("0");
  await page.getByRole("button", { name: "Resume", exact: true }).click();
  await page.clock.runFor(10000);
  expect(await page.evaluate(() => window.chamberProbe.transports.filter((track) => track.mass < 0.001).length)).toBe(0);
  await page.locator(".playground-panel > summary").click();
  await tapChamber(page);
  const electrons = await page.evaluate(() => window.chamberProbe.transports.filter((track) => track.mass < 0.001));
  expect(electrons).toHaveLength(2);
  for (const track of electrons) {
    expect(track.start.x).toBeCloseTo(page.viewportSize().width * 0.75, 0);
    expect(track.start.y).toBeCloseTo(page.viewportSize().height * 0.8, 0);
  }
});


test("busy activity shortens trails and repeated clicks respect the track budget", async ({ page }) => {
  await seedChamber(page);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.clock.install();
  await page.goto("/playground/");
  await observeChamber(page);
  await page.evaluate(() => {
    window.chamberFills = 0;
    const fill = CanvasRenderingContext2D.prototype.fill;
    CanvasRenderingContext2D.prototype.fill = function (...args) {
      window.chamberFills += 1;
      return fill.apply(this, args);
    };
  });
  await page.selectOption("#playground-topology", "dalitz");
  await page.locator("#playground-event-rate").fill("0");
  await page.locator(".playground-beam > summary").click();
  await page.locator("#playground-beam-density").fill("1");
  await page.getByRole("button", { name: "Resume", exact: true }).click();
  await page.clock.runFor(1500);
  await finishBeam(page);
  const quietHold = await page.evaluate(() => window.chamberProbe.hold);
  await page.locator("#playground-event-rate").fill("10");
  expect(await page.evaluate(() => window.chamberProbe.hold)).toBeLessThan(quietHold);
  await page.locator("#playground-beam-density").fill("60");
  await page.getByRole("button", { name: "Clear", exact: true }).click();
  await page.clock.runFor(1500);
  await finishBeam(page);
  await page.locator(".playground-panel > summary").click();
  await page.evaluate(() => {
    for (let index = 0; index < 50; index += 1) document.body.dispatchEvent(new MouseEvent("click", {
      bubbles: true, button: 0, clientX: innerWidth * 0.75, clientY: innerHeight * 0.8,
    }));
  });
  await finishBeam(page);
  const probe = await page.evaluate(() => window.chamberProbe);
  expect(probe.active).toBeGreaterThan(60);
  expect(probe.maxActive).toBeLessThanOrEqual(96);
  expect(probe.beams).toBe(1);
  await page.evaluate(() => { window.chamberFills = 0; });
  await page.clock.runFor(500);
  expect(await page.evaluate(() => window.chamberFills)).toBeLessThan(10000);
});


test("random beam angles change per passage and restore the manual aim", async ({ page }) => {
  await seedChamber(page, 7);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.clock.install();
  await page.goto("/playground/");
  await observeChamber(page);
  await page.locator("#playground-event-rate").fill("0");
  await page.locator(".playground-beam > summary").click();
  await page.locator("#playground-beam-density").fill("1");
  await page.locator("#playground-beam-direction").fill("90");
  await page.getByRole("switch", { name: "Random angle", exact: true }).check();
  await expect(page.locator("#playground-beam-direction")).toBeDisabled();
  await page.selectOption("#playground-view", "photograph");
  await page.clock.runFor(500);
  const beams = () => page.evaluate(() => window.chamberProbe.transports.filter((track) => track.role === "beam-incoming"));
  const direction = (track) => (Math.atan2(-(track.points[1].y - track.points[0].y), track.points[1].x - track.points[0].x) * 180 / Math.PI + 360) % 360;
  const error = (a, b) => Math.abs((a - b + 540) % 360 - 180);
  await finishBeam(page);
  const first = (await beams()).at(-1);
  expect(first).toBeTruthy();
  const firstAngle = Number.parseFloat(await page.locator("#playground-beam-direction-value").textContent());
  expect(error(direction(first), firstAngle)).toBeLessThan(1);
  await page.getByRole("button", { name: "Next photo", exact: true }).click();
  await page.clock.runFor(500);
  await finishBeam(page);
  const secondAngle = Number.parseFloat(await page.locator("#playground-beam-direction-value").textContent());
  expect(secondAngle).not.toBe(firstAngle);
  expect(error(direction((await beams()).at(-1)), secondAngle)).toBeLessThan(1);
  const held = await chamberImage(page);
  await page.getByRole("switch", { name: "Random angle", exact: true }).uncheck();
  await expect(page.locator("#playground-beam-direction")).toBeEnabled();
  await expect(page.locator("#playground-beam-direction")).toHaveValue("90");
  expect(await chamberImage(page)).toBe(held);
  await page.getByRole("button", { name: "Next photo", exact: true }).click();
  await page.clock.runFor(500);
  await finishBeam(page);
  expect(error(direction((await beams()).at(-1)), 90)).toBeLessThan(1);
});


for (const kind of ["track", "decay", "cascade", "dalitz"]) {
  test(`beam-target ${kind} consumes the primary at a conserved production vertex`, async ({ page }) => {
    await seedChamber(page, 19);
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.clock.install();
    await page.goto("/playground/");
    await observeChamber(page);
    await page.selectOption("#playground-topology", kind);
    await page.locator("#playground-event-rate").fill("10");
    await page.locator(".playground-beam > summary").click();
    await page.locator("#playground-beam-density").fill("1");
    await page.selectOption("#playground-view", "photograph");
    await page.clock.runFor(1500);
    await finishBeam(page);
    const tracks = await page.evaluate(() => window.chamberProbe.transports);
    const incoming = tracks.find((track) => track.role === "beam-incoming");
    expect(incoming.stopReason).toBe("interaction");
    const vertex = incoming.interaction;
    expect(vertex.kind).toBe(kind);
    expect(incoming.points.at(-1)).toEqual(vertex.vertex);
    expect(incoming.delay + incoming.times.at(-1)).toBe(vertex.time);
    expect(incoming.endState.px).toBe(vertex.initial[0].px);
    expect(incoming.endState.py).toBe(vertex.initial[0].py);
    expect(vertex.initial[1].id).toBe("p");
    for (const key of ["energy", "px", "py"]) {
      const total = (list) => list.reduce((sum, item) => sum + item[key], 0);
      expect(Math.abs(total(vertex.initial) - total(vertex.products))).toBeLessThan(1e-8);
    }
    if (kind === "track" || kind === "dalitz") {
      const recoil = tracks.find((track) => track.role === "recoil");
      expect(recoil).toBeTruthy();
      expect(recoil.start).toEqual(vertex.vertex);
    }
    expect(tracks.some((track) => track.role === "secondary")).toBe(true);
  });
}

test("the beam bunch has longitudinal spread rather than a shared arrival front", async ({ page }) => {
  await seedChamber(page, 77);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.clock.install();
  await page.goto("/playground/");
  await observeChamber(page);
  await page.locator("#playground-event-rate").fill("0");
  await page.locator(".playground-beam > summary").click();
  await page.locator("#playground-beam-density").fill("20");
  await page.getByRole("button", { name: "Resume", exact: true }).click();
  await page.clock.runFor(1500);
  await finishBeam(page);
  const arrivals = await page.evaluate(() => window.chamberProbe.transports
    .filter((track) => track.role === "beam-incoming").map((track) => {
      const index = track.points.findIndex((point) => point.x >= 0 && point.x <= innerWidth && point.y >= 0 && point.y <= innerHeight);
      return track.delay + track.times[index];
    }));
  expect(arrivals).toHaveLength(20);
  expect(Math.max(...arrivals) - Math.min(...arrivals)).toBeGreaterThan(100);
});
