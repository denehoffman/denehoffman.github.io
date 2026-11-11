const { test } = require("node:test");
const assert = require("node:assert/strict");
const physics = require("../static/bubble-physics.js");
const muon = { mass: 0.105658376, charge: 1 };
const proton = { mass: 0.938272088, charge: 1 };
const electron = { mass: 0.000510999, charge: -1 };

function randomSeed(seed) {
  return () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) + 0.5) / 4294967296;
}

function trajectory(particle, momentum, options = {}) {
  return physics.transport(particle, { x: 0, y: 0 }, { px: momentum, py: 0 }, {
    pixelsPerMeter: 1, stepPixels: 0.01, maximumPathMeters: 1,
    maximumTransportSteps: 20000, random: randomSeed(42), ...options,
  });
}

// Independent numerical reference points from the PDG liquid-hydrogen muon
// table: p[MeV/c], collision stopping power [MeV cm²/g]. Not fitted to our code.
test("hydrogen stopping agrees with PDG muon reference points", () => {
  for (const [momentum, reference] of [[0.04704, 16.51], [0.1764, 4.568], [0.4372, 4.034]]) {
    const expected = reference * 0.0708 * 100 / 1000;
    const actual = physics.stoppingPower(muon, momentum);
    assert.ok(Math.abs(actual / expected - 1) < 0.05, `${momentum}: ${actual} vs ${expected}`);
  }
  assert.ok(physics.stoppingPower(proton, 0.15) > physics.stoppingPower(proton, 1));
  assert.ok(physics.stoppingPower(electron, 0.01) > 0.015);
  assert.ok(physics.stoppingPower(electron, 0.01) < 0.06);
});

test("magnetic transport preserves energy and the expected circle without losses or scattering", () => {
  const result = trajectory(muon, 0.3, { scattering: false, stopping: false, maximumPathMeters: 2 });
  const radius = 0.3 / (0.299792458 * 1.4);
  for (const state of result.states) {
    assert.ok(Math.abs(Math.hypot(state.px, state.py) - 0.3) < 1e-12);
    assert.ok(Math.abs(Math.hypot(state.x, state.y - radius) - radius) < 1e-10);
  }
  const negative = trajectory({ ...muon, charge: -1 }, 0.3, { scattering: false, stopping: false, maximumPathMeters: 2 });
  assert.ok(Math.abs(negative.points.at(-1).y + result.points.at(-1).y) < 1e-10);
});

test("stopping deposits the lost energy and range converges when steps are halved", () => {
  const coarse = trajectory(proton, 0.3, { scattering: false, maximumPathMeters: 10 });
  const fine = trajectory(proton, 0.3, { scattering: false, stepPixels: 0.005, maximumPathMeters: 10 });
  assert.equal(coarse.stopReason, "ranged-out");
  assert.equal(fine.stopReason, "ranged-out");
  assert.ok(Math.abs(coarse.states.at(-1).pathLength / fine.states.at(-1).pathLength - 1) < 0.025);
  let deposited = 0;
  for (let index = 1; index < fine.states.length; index += 1) {
    const previous = fine.states[index - 1];
    const current = fine.states[index];
    const oldEnergy = Math.hypot(proton.mass, previous.px, previous.py);
    const energy = Math.hypot(proton.mass, current.px, current.py);
    assert.ok(energy <= oldEnergy + 1e-12);
    deposited += previous.stoppingPower * (current.pathLength - previous.pathLength);
  }
  const final = fine.states.at(-1);
  assert.ok(Math.abs(deposited - (Math.hypot(proton.mass, 0.3) - Math.hypot(proton.mass, final.px, final.py))) < 1e-10);
});

function variance(values) {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  return values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
}

test("scattering ensembles keep their physical spread when the step size changes", () => {
  const results = [];
  for (const stepPixels of [0.02, 0.01]) {
    const angles = [];
    for (let seed = 1; seed <= 600; seed += 1) {
      const result = trajectory(muon, 0.3, { stepPixels, magneticFieldTesla: 0,
        stopping: false, random: randomSeed(seed) });
      const final = result.states.at(-1);
      angles.push(Math.atan2(final.py, final.px));
    }
    results.push(variance(angles));
  }
  const expected = (0.0136 / (0.3 * (0.3 / Math.hypot(muon.mass, 0.3)))) ** 2 / 8.905;
  for (const result of results) assert.ok(Math.abs(result / expected - 1) < 0.16);
  assert.ok(Math.abs(results[0] / results[1] - 1) < 0.2);
});

function straightTrack(segments, ionization = physics.HYDROGEN.minimumIonizationGeVPerMeter) {
  return { delay: 0, points: Array.from({ length: segments + 1 }, (_, i) => ({ x: 360 * i / segments, y: 0 })),
    times: Array.from({ length: segments + 1 }, (_, i) => 800 * i / segments),
    states: Array.from({ length: segments + 1 }, (_, i) => ({ pathLength: i / segments, stoppingPower: ionization })) };
}

test("bubble sampling is independent of line segmentation and responds to ionization", () => {
  const sample = (segments, loss) => physics.sampleBubbles(straightTrack(segments, loss), { random: randomSeed(77) });
  const coarse = sample(1);
  const fine = sample(100);
  assert.equal(coarse.length, fine.length);
  assert.ok(coarse.length > 200 && coarse.length < 330);
  for (let i = 0; i < coarse.length; i += 1) {
    assert.ok(Math.abs(coarse[i].x - fine[i].x) < 1e-8);
    assert.ok(Math.abs(coarse[i].time - fine[i].time) < 1e-8);
    assert.ok(coarse[i].radius > 0 && Number.isFinite(coarse[i].x));
  }
  assert.ok(sample(100, 4 * physics.HYDROGEN.minimumIonizationGeVPerMeter).length > fine.length * 3);
});


test("beam entry reaches the chamber from upstream throughout a full turn", () => {
  for (const [width, height] of [[1280, 800], [412, 915]]) {
    for (let degrees = 0; degrees < 360; degrees += 15) {
      const angle = -degrees * Math.PI / 180;
      for (const offset of [-40, 0, 40]) {
        const entry = physics.beamEntry(width, height, angle, offset);
        assert.ok(entry.x < 0 || entry.x > width || entry.y < 0 || entry.y > height);
        const dx = Math.cos(angle), dy = Math.sin(angle);
        const targetX = width / 2 - dy * offset;
        const targetY = height / 2 + dx * offset;
        const along = (targetX - entry.x) * dx + (targetY - entry.y) * dy;
        const across = (targetX - entry.x) * dy - (targetY - entry.y) * dx;
        assert.ok(along > 0 && Math.abs(across) < 1e-9);
      }
    }
  }
  assert.ok(physics.beamEntry(1280, 800, 0).x < 0);
  assert.ok(physics.beamEntry(1280, 800, -Math.PI / 2).y > 800);
  assert.ok(physics.beamEntry(1280, 800, -Math.PI).x > 1280);
  assert.ok(physics.beamEntry(1280, 800, -3 * Math.PI / 2).y < 0);
});


const interactions = require("../static/bubble-interactions.js");
function conserved(initial, final) {
  for (const key of ["energy", "px", "py"]) {
    const sum = (list) => list.reduce((value, item) => value + item[key], 0);
    assert.ok(Math.abs(sum(initial) - sum(final)) < 1e-8, key);
  }
  for (const product of final) {
    assert.ok(product.energy >= interactions.particles[product.id].mass);
    assert.ok(Math.abs(product.energy ** 2 - product.px ** 2 - product.py ** 2
      - interactions.particles[product.id].mass ** 2) < 1e-8);
  }
}

test("pion-hydrogen channels conserve four-momentum and strong quantum numbers", () => {
  for (const kind of ["track", "decay", "cascade", "dalitz", "mixed"]) {
    for (let seed = 1; seed <= 100; seed += 1) {
      const angle = seed * 0.17;
      const momentum = 5 + seed / 25;
      const event = interactions.produce(kind, { px: momentum * Math.cos(angle), py: momentum * Math.sin(angle) }, randomSeed(seed));
      assert.ok(event);
      conserved(event.initial, event.products);
      assert.equal(event.initial[1].id, "p");
      assert.equal(event.initial[1].px, 0);
      assert.equal(event.initial[1].py, 0);
      for (const [key, expected] of [["charge", 0], ["baryon", 1], ["strangeness", 0]]) {
        assert.equal(event.products.reduce((sum, item) => sum + interactions.particles[item.id][key], 0), expected);
      }
    }
  }
});

test("production respects laboratory thresholds rather than injecting missing energy", () => {
  const mBeam = 0.13957039, mTarget = 0.938272088;
  for (const [kind, totalMass] of [["decay", 0.497611 + 1.115683],
      ["cascade", 0.493677 + 0.497611 + 1.32171], ["dalitz", mBeam + mTarget + 0.1349768]]) {
    const thresholdEnergy = (totalMass ** 2 - mBeam ** 2 - mTarget ** 2) / (2 * mTarget);
    const thresholdMomentum = Math.sqrt(thresholdEnergy ** 2 - mBeam ** 2);
    assert.equal(interactions.produce(kind, { px: thresholdMomentum - 1e-7, py: 0 }, randomSeed(4)), null);
    const above = interactions.produce(kind, { px: thresholdMomentum + 1e-5, py: 0 }, randomSeed(4));
    assert.ok(above);
    conserved(above.initial, above.products);
  }
});

test("forward elastic scattering returns the beam and a stationary target", () => {
  const event = interactions.produce("track", { px: 5, py: 0 }, () => 0);
  assert.ok(Math.abs(event.products[0].px - 5) < 1e-12);
  assert.ok(Math.abs(event.products[1].px) < 1e-12);
  assert.ok(Math.abs(event.products[1].energy - 0.938272088) < 1e-12);
  const recoil = interactions.produce("track", { px: 5, py: 0 }, () => 0.5);
  assert.ok(recoil.products[1].energy > 0.938272088 + 1);
  conserved(recoil.initial, recoil.products);
});

test("selected secondary decay chains conserve four-momentum at the actual parent momentum", () => {
  for (const id of ["K0", "Lambda", "Xi-", "pi0"]) {
    for (let seed = 1; seed <= 150; seed += 1) {
      const event = interactions.decay(id, { px: seed / 30, py: -seed / 70 }, randomSeed(seed));
      conserved([event.parent], event.products);
      for (const key of ["charge", "baryon"]) {
        assert.equal(event.products.reduce((sum, item) => sum + interactions.particles[item.id][key], 0), interactions.particles[id][key]);
      }
      for (const product of event.products) {
        const child = interactions.decay(product.id, product, randomSeed(seed + 1));
        if (child) conserved([child.parent], child.products);
      }
    }
  }
});
