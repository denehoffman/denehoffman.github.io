(() => {
  "use strict";

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const playground = document.querySelector("[data-particle-playground]");
  if (reducedMotion.matches && !playground) return;
  if (playground) document.body.classList.add("is-particle-playground");
  const physics = window.BubblePhysics;
  const interactions = window.BubbleInteractions;
  const settings = { speed: 1, automatic: true, activity: playground ? 5 : 0.6, replaySpeed: 1,
    mode: "replay", beamAutomatic: true, randomBeamAngle: false, beamDensity: playground ? 10 : 2, backgroundTracks: playground ? 0.6 : 0.25, specks: playground ? 0.4 : 0.16 };
  const EXPOSURE = { growth: 1000, hold: 7000 };
  const TRACK_BUDGET = 96;
  let exposureStarted = 0;
  let exposureInitialized = false;
  let nextBackgroundAt = 2000;
  let nextBeamAt = 300;
  const BEAM_INTERVAL = playground ? 6000 : 10000;
  const MAXIMUM_BEAM_TRACKS = 120;
  let specks = [];
  let sceneRevision = 0;
  let heldPhotoKey = null;
  let simulationTime = 0;
  let previousFrame = 0;

  // Visual handles. Particle masses, charges, and cτ values below remain in
  // physical units; these constants map the simulated chamber onto the page.
  const CONFIG = {
    magneticFieldTesla: 1.4,
    pixelsPerMeter: 360,
    stepPixels: 3.5,
    visualMillisecondsPerMeterAtC: 800,
    lifetimeScale: 8,
    tailHoldMilliseconds: 9000,
    tailFadeMilliseconds: 4000,
    maximumTransportSteps: 14000,
    maximumConcurrentEvents: 4,
  };
  const PAUSE_STORAGE_KEY = "bubble-chamber-paused";

  const PARTICLES = {
    "pi+": { name: "π+", mass: 0.13957039, charge: 1, cTau: 7.804 },
    "pi-": { name: "π−", mass: 0.13957039, charge: -1, cTau: 7.804 },
    "pi0": { name: "π0", mass: 0.1349768, charge: 0, cTau: 0.000000025 },
    "mu+": { name: "μ+", mass: 0.105658376, charge: 1, cTau: 658.65 },
    "mu-": { name: "μ−", mass: 0.105658376, charge: -1, cTau: 658.65 },
    "e+": { name: "e+", mass: 0.000510999, charge: 1, cTau: Infinity },
    "e-": { name: "e−", mass: 0.000510999, charge: -1, cTau: Infinity },
    gamma: { name: "γ", mass: 0, charge: 0, cTau: Infinity },
    nu: { name: "ν", mass: 0, charge: 0, cTau: Infinity },
    p: { name: "p", mass: 0.938272088, charge: 1, cTau: Infinity },
    pbar: { name: "p̄", mass: 0.938272088, charge: -1, cTau: Infinity },
    n: { name: "n", mass: 0.939565421, charge: 0, cTau: Infinity },
    nbar: { name: "n̄", mass: 0.939565421, charge: 0, cTau: Infinity },
    KS: {
      name: "K⁰S",
      mass: 0.497611,
      charge: 0,
      cTau: 0.02684,
      channels: [
        { weight: 0.692, daughters: ["pi+", "pi-"] },
        { weight: 0.3069, daughters: ["pi0", "pi0"] },
        { weight: 0.000357, daughters: ["pi-", "e+", "nu"] },
        { weight: 0.000357, daughters: ["pi+", "e-", "nu"] },
        { weight: 0.000228, daughters: ["pi-", "mu+", "nu"] },
        { weight: 0.000228, daughters: ["pi+", "mu-", "nu"] },
      ],
    },
    Lambda: {
      name: "Λ",
      mass: 1.115683,
      charge: 0,
      cTau: 0.0789,
      channels: [
        { weight: 0.640368, daughters: ["p", "pi-"] },
        { weight: 0.358647, daughters: ["n", "pi0"] },
        { weight: 0.000834, daughters: ["p", "e-", "nu"] },
        { weight: 0.000151, daughters: ["p", "mu-", "nu"] },
      ],
    },
    LambdaBar: {
      name: "Λ̄",
      mass: 1.115683,
      charge: 0,
      cTau: 0.0789,
      channels: [
        { weight: 0.640368, daughters: ["pbar", "pi+"] },
        { weight: 0.358647, daughters: ["nbar", "pi0"] },
        { weight: 0.000834, daughters: ["pbar", "e+", "nu"] },
        { weight: 0.000151, daughters: ["pbar", "mu+", "nu"] },
      ],
    },
    "K+": {
      name: "K+",
      mass: 0.493677,
      charge: 1,
      cTau: 3.713,
      channels: [
        { weight: 0.6356, daughters: ["mu+", "nu"] },
        { weight: 0.2067, daughters: ["pi+", "pi0"] },
        { weight: 0.05583, daughters: ["pi+", "pi+", "pi-"] },
        { weight: 0.0507, daughters: ["pi0", "e+", "nu"] },
        { weight: 0.03352, daughters: ["pi0", "mu+", "nu"] },
        { weight: 0.0176, daughters: ["pi+", "pi0", "pi0"] },
      ],
    },
    "K-": {
      name: "K−",
      mass: 0.493677,
      charge: -1,
      cTau: 3.713,
      channels: [
        { weight: 0.6356, daughters: ["mu-", "nu"] },
        { weight: 0.2067, daughters: ["pi-", "pi0"] },
        { weight: 0.05583, daughters: ["pi-", "pi-", "pi+"] },
        { weight: 0.0507, daughters: ["pi0", "e-", "nu"] },
        { weight: 0.03352, daughters: ["pi0", "mu-", "nu"] },
        { weight: 0.0176, daughters: ["pi-", "pi0", "pi0"] },
      ],
    },
    "Xi-": {
      name: "Ξ−",
      mass: 1.32171,
      charge: -1,
      cTau: 0.0491,
      channels: [{ weight: 1, daughters: ["Lambda", "pi-"] }],
    },
    XiBar: {
      name: "Ξ̄+",
      mass: 1.32171,
      charge: 1,
      cTau: 0.0491,
      channels: [{ weight: 1, daughters: ["LambdaBar", "pi+"] }],
    },
  };

  const canvas = document.createElement("canvas");
  canvas.className = "bubble-chamber";
  canvas.setAttribute("aria-hidden", "true");
  document.body.prepend(canvas);

  const context = canvas.getContext("2d");
  if (!context) {
    canvas.remove();
    return;
  }

  const controls = document.createElement("aside");
  controls.className = "bubble-chamber-controls";
  controls.setAttribute("aria-label", "Bubble chamber controls");
  controls.innerHTML = `
    <p class="bubble-chamber-status" role="status" aria-live="polite"></p>
    <button class="bubble-chamber-control bubble-chamber-pause" type="button">
      <span aria-hidden="true">Ⅱ</span>
    </button>
    <details class="bubble-chamber-info">
      <summary class="bubble-chamber-control" aria-label="About the bubble chamber">
        <span aria-hidden="true">?</span>
      </summary>
      <section class="bubble-chamber-explainer">
        <p class="bubble-chamber-explainer__kicker">Background simulation</p>
        <h2>Reading the chamber</h2>
        <p>These tracks are simulated particle paths in liquid hydrogen. Small deflections model scattering in the liquid; a directed particle beam crosses the chamber, while unrelated cosmic tracks and chamber specks form the background.</p>
        <ul>
          <li><span class="bubble-chamber-key bubble-chamber-key--curve" aria-hidden="true"></span><span>Charged particles curve in a magnetic field. For particles with the same charge magnitude, tighter curves indicate lower momentum.</span></li>
          <li><span class="bubble-chamber-key bubble-chamber-key--vertex" aria-hidden="true"></span><span>A decay can produce a kink, a fork, or a V where new charged tracks begin. Neutral particles leave no track.</span></li>
          <li><span class="bubble-chamber-key bubble-chamber-key--drops" aria-hidden="true"></span><span>Each track is made of individual bubbles. Denser tracks indicate greater energy deposition in the liquid.</span></li>
        </ul>
        <p class="bubble-chamber-explainer__hint">Choose an event type in the explorer, then click or tap the page to create an event. The default is a π⁰ Dalitz decay.</p>
        <p class="bubble-chamber-explainer__reference"><a href="https://home.cern/science/experiments/gargamelle/">Learn about CERN’s Gargamelle bubble chamber <span aria-hidden="true">↗</span></a></p>
        <p class="bubble-chamber-playground-link"><a href="/playground/">Open particle playground <span aria-hidden="true">↗</span></a></p>
      </section>
    </details>
    <details class="bubble-chamber-explorer">
      <summary class="bubble-chamber-control" aria-label="Open the event explorer">
        <span aria-hidden="true">⌁</span>
      </summary>
      <section class="bubble-chamber-explorer-panel">
        <p class="bubble-chamber-explainer__kicker">Event explorer</p>
        <h2>Explore particle tracks</h2>
        <p>Choose an event to see its particles and compare the reaction with the tracks in the chamber.</p>
        <div class="bubble-chamber-event-types" role="group" aria-label="Event type">
          <button type="button" data-event-kind="decay"><span class="bubble-chamber-topology" aria-hidden="true">V</span>Single decay</button>
          <button type="button" data-event-kind="cascade"><span class="bubble-chamber-topology" aria-hidden="true">Y</span>Cascade decay</button>
          <button type="button" data-event-kind="dalitz"><span class="bubble-chamber-topology" aria-hidden="true">γ*</span>Dalitz decay</button>
          <button type="button" data-event-kind="track"><span class="bubble-chamber-topology" aria-hidden="true">↝</span>Single track</button>
        </div>
        <article class="bubble-chamber-event-readout" aria-live="polite">
          <p class="bubble-chamber-event-readout__empty">Choose an event type to see its particles.</p>
        </article>
        <p class="bubble-chamber-explorer-note"><span aria-hidden="true">○</span> Neutral particles leave no track. Positive and negative particles bend in opposite directions.</p>
        <p class="bubble-chamber-playground-link"><a href="/playground/">Open particle playground <span aria-hidden="true">↗</span></a></p>
      </section>
    </details>
  `;
  document.body.append(controls);

  const pauseButton = controls.querySelector(".bubble-chamber-pause");
  const infoPanel = controls.querySelector(".bubble-chamber-info");
  const explorer = controls.querySelector(".bubble-chamber-explorer");
  const eventReadout = controls.querySelector(".bubble-chamber-event-readout");
  const status = controls.querySelector(".bubble-chamber-status");

  const TOPOLOGY_LATEX = {
    decay: "X \\to a+b",
    cascade: "\\Xi \\to \\Lambda\\pi,\\; \\Lambda \\to p\\pi",
    dalitz: "\\pi^0 \\to \\gamma e^+e^-",
    track: "q^{\\pm}",
  };

  const events = [];
  const beamJobs = new Set();
  function removeEvent(index) {
    events[index].layer.dispose();
    events.splice(index, 1);
    sceneRevision += 1;
  }
  function cancelBeams() {
    for (const job of beamJobs) job.cancelled = true;
  }
  function clearEvents() {
    cancelBeams();
    while (events.length) removeEvent(events.length - 1);
  }
  let width = 0;
  let height = 0;
  let pixelRatio = 1;
  let animationFrame = 0;
  let paused = playground ? reducedMotion.matches : readPausedPreference();
  let statusTimeout = 0;
  let selectedEventKind = "dalitz";
  let beamAngle = 0;
  let manualBeamDegrees = 0;

  function readPausedPreference() {
    try {
      return window.localStorage.getItem(PAUSE_STORAGE_KEY) === "true";
    } catch {
      return false;
    }
  }

  function storePausedPreference(value) {
    try {
      window.localStorage.setItem(PAUSE_STORAGE_KEY, String(value));
    } catch {
      // Storage can be unavailable in restricted browsing modes.
    }
  }

  function randomBetween(minimum, maximum) {
    return minimum + Math.random() * (maximum - minimum);
  }

  function logUniform(minimum, maximum) {
    return Math.exp(randomBetween(Math.log(minimum), Math.log(maximum)));
  }

  function singleTrackMomentum() {
    const choice = Math.random();
    if (choice < 0.52) return logUniform(1.4, 4.2);
    if (choice < 0.82) return logUniform(0.38, 1.4);
    return logUniform(0.07, 0.22);
  }

  function chooseWeighted(items) {
    const total = items.reduce((sum, item) => sum + item.weight, 0);
    let choice = Math.random() * total;
    for (const item of items) {
      choice -= item.weight;
      if (choice <= 0) return item;
    }
    return items[items.length - 1];
  }

  function resize() {
    cancelBeams();
    width = window.innerWidth;
    height = window.innerHeight;
    pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * pixelRatio);
    canvas.height = Math.round(height * pixelRatio);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    if (!playground) clearEvents();
    buildChamberSpecks();
    draw(simulationTime);
  }

  function isOnScreen(point, margin = 0) {
    return (
      point.x >= -margin &&
      point.x <= width + margin &&
      point.y >= -margin &&
      point.y <= height + margin
    );
  }

  function inwardStart() {
    const margin = 18;
    const edge = Math.floor(Math.random() * 4);
    const drift = randomBetween(-0.48, 0.48);
    if (edge === 0) return { x: -margin, y: randomBetween(0, height), angle: drift };
    if (edge === 1) {
      return { x: width + margin, y: randomBetween(0, height), angle: Math.PI + drift };
    }
    if (edge === 2) {
      return { x: randomBetween(0, width), y: -margin, angle: Math.PI / 2 + drift };
    }
    return {
      x: randomBetween(0, width),
      y: height + margin,
      angle: -Math.PI / 2 + drift,
    };
  }

  function beamStart() {
    return physics.beamEntry(width, height, beamAngle,
      randomBetween(-0.3, 0.3) * Math.min(width, height));
  }

  function clickStart(point) {
    const x = Math.max(0, Math.min(width, point.x));
    const y = Math.max(0, Math.min(height, point.y));
    const centerAngle = Math.atan2(height / 2 - y, width / 2 - x);
    return {
      x,
      y,
      angle: Number.isFinite(point.angle) ? point.angle : centerAngle + randomBetween(-0.65, 0.65),
    };
  }

  function momentumFromMagnitude(magnitude, angle) {
    return { px: magnitude * Math.cos(angle), py: magnitude * Math.sin(angle) };
  }

  function momentumMagnitude(momentum) {
    return Math.hypot(momentum.px, momentum.py);
  }

  function energy(particle, momentum) {
    return Math.hypot(particle.mass, momentumMagnitude(momentum));
  }

  function boostFourVector(restEnergy, restMomentum, parentMomentum, parentMass) {
    const parentEnergy = Math.hypot(parentMass, momentumMagnitude(parentMomentum));
    const betaX = parentMomentum.px / parentEnergy;
    const betaY = parentMomentum.py / parentEnergy;
    const betaSquared = betaX * betaX + betaY * betaY;
    if (betaSquared < 1e-14) {
      return { energy: restEnergy, ...restMomentum };
    }
    const gamma = parentEnergy / parentMass;
    const betaDotP = betaX * restMomentum.px + betaY * restMomentum.py;
    const factor = ((gamma - 1) * betaDotP) / betaSquared + gamma * restEnergy;
    return {
      energy: gamma * (restEnergy + betaDotP),
      px: restMomentum.px + factor * betaX,
      py: restMomentum.py + factor * betaY,
    };
  }

  function twoBodyDecay(parent, parentMomentum, first, second) {
    const parentMassSquared = parent.mass * parent.mass;
    const sumMasses = first.mass + second.mass;
    const massDifference = first.mass - second.mass;
    const kallenProduct =
      (parentMassSquared - sumMasses * sumMasses) *
      (parentMassSquared - massDifference * massDifference);
    const restMomentumMagnitude = Math.sqrt(Math.max(0, kallenProduct)) / (2 * parent.mass);
    const direction = randomBetween(0, Math.PI * 2);
    const firstRestMomentum = momentumFromMagnitude(restMomentumMagnitude, direction);
    const secondRestMomentum = { px: -firstRestMomentum.px, py: -firstRestMomentum.py };
    const firstLab = boostFourVector(
      Math.hypot(first.mass, restMomentumMagnitude),
      firstRestMomentum,
      parentMomentum,
      parent.mass,
    );
    const secondLab = boostFourVector(
      Math.hypot(second.mass, restMomentumMagnitude),
      secondRestMomentum,
      parentMomentum,
      parent.mass,
    );

    // Guard the simulator's central invariant against future catalog changes.
    const parentEnergy = Math.hypot(parent.mass, momentumMagnitude(parentMomentum));
    const residual = Math.max(
      Math.abs(firstLab.energy + secondLab.energy - parentEnergy),
      Math.abs(firstLab.px + secondLab.px - parentMomentum.px),
      Math.abs(firstLab.py + secondLab.py - parentMomentum.py),
    );
    if (residual > 1e-9) throw new Error("Bubble-chamber decay failed four-momentum conservation");

    return [firstLab, secondLab];
  }

  function restFrameMomentum(parentMass, firstMass, secondMass) {
    const parentMassSquared = parentMass * parentMass;
    const sum = firstMass + secondMass;
    const difference = firstMass - secondMass;
    return (
      Math.sqrt(
        Math.max(
          0,
          (parentMassSquared - sum * sum) *
            (parentMassSquared - difference * difference),
        ),
      ) /
      (2 * parentMass)
    );
  }

  function threeBodyDecay(parent, parentMomentum, first, second, third) {
    const minimumMass = first.mass + second.mass;
    const maximumMass = parent.mass - third.mass;
    let maximumWeight = 0;
    for (let index = 0; index <= 80; index += 1) {
      const mass = minimumMass + ((maximumMass - minimumMass) * index) / 80;
      maximumWeight = Math.max(
        maximumWeight,
        restFrameMomentum(parent.mass, mass, third.mass) *
          restFrameMomentum(mass, first.mass, second.mass),
      );
    }

    let intermediateMass = minimumMass;
    for (let attempt = 0; attempt < 200; attempt += 1) {
      const candidate = randomBetween(minimumMass, maximumMass);
      const weight =
        restFrameMomentum(parent.mass, candidate, third.mass) *
        restFrameMomentum(candidate, first.mass, second.mass);
      if (Math.random() * maximumWeight <= weight) {
        intermediateMass = candidate;
        break;
      }
    }

    const intermediate = {
      name: "virtual",
      mass: intermediateMass,
      charge: first.charge + second.charge,
      cTau: 0,
    };
    const [intermediateVector, thirdVector] = twoBodyDecay(
      parent,
      parentMomentum,
      intermediate,
      third,
    );
    const [firstVector, secondVector] = twoBodyDecay(
      intermediate,
      { px: intermediateVector.px, py: intermediateVector.py },
      first,
      second,
    );

    const parentEnergy = Math.hypot(parent.mass, momentumMagnitude(parentMomentum));
    const residual = Math.max(
      Math.abs(firstVector.energy + secondVector.energy + thirdVector.energy - parentEnergy),
      Math.abs(firstVector.px + secondVector.px + thirdVector.px - parentMomentum.px),
      Math.abs(firstVector.py + secondVector.py + thirdVector.py - parentMomentum.py),
    );
    if (residual > 1e-9) {
      throw new Error("Bubble-chamber three-body decay failed four-momentum conservation");
    }
    return [firstVector, secondVector, thirdVector];
  }

  function transportCharged(particle, start, initialMomentum) {
    return physics.transport(particle, start, initialMomentum, {
      ...CONFIG, isInside: isOnScreen,
    });
  }

  function visibleIndexBounds(transport) {
    let first = -1;
    let last = -1;
    for (let index = 0; index < transport.points.length; index += 1) {
      if (!isOnScreen(transport.points[index], 4)) continue;
      if (first < 0) first = index;
      last = index;
    }
    return { first, last };
  }

  function sampleConditionedExponential(mean, minimum, maximum) {
    const span = maximum - minimum;
    if (span <= 0) return minimum;
    const truncatedMass = -Math.expm1(-span / mean);
    return minimum - mean * Math.log1p(-Math.random() * truncatedMass);
  }

  function selectChargedDecayState(parent, transport) {
    const { first, last } = visibleIndexBounds(transport);
    if (first < 0 || last - first < 12) return null;
    const minimumIndex = Math.min(last, first + 6);
    const maximumIndex = Math.max(minimumIndex, last - 6);
    const minimum = transport.states[minimumIndex].properLength;
    const maximum = transport.states[maximumIndex].properLength;
    const target = sampleConditionedExponential(
      parent.cTau * CONFIG.lifetimeScale,
      minimum,
      maximum,
    );
    let index = minimumIndex;
    while (index < maximumIndex && transport.states[index].properLength < target) index += 1;
    return index;
  }

  function rayViewportInterval(start, angle) {
    const dx = Math.cos(angle);
    const dy = Math.sin(angle);
    let minimum = -Infinity;
    let maximum = Infinity;
    for (const [origin, direction, low, high] of [
      [start.x, dx, 0, width],
      [start.y, dy, 0, height],
    ]) {
      if (Math.abs(direction) < 1e-9) {
        if (origin < low || origin > high) return null;
        continue;
      }
      const first = (low - origin) / direction;
      const second = (high - origin) / direction;
      minimum = Math.max(minimum, Math.min(first, second));
      maximum = Math.min(maximum, Math.max(first, second));
    }
    if (maximum <= Math.max(0, minimum)) return null;
    return { enterPixels: Math.max(0, minimum), exitPixels: maximum };
  }

  function selectNeutralDecay(parent, start, angle, momentumMagnitudeGeV) {
    const interval = rayViewportInterval(start, angle);
    if (!interval) return null;
    const marginMeters = 18 / CONFIG.pixelsPerMeter;
    const minimumMeters = interval.enterPixels / CONFIG.pixelsPerMeter + marginMeters;
    const maximumMeters = interval.exitPixels / CONFIG.pixelsPerMeter - marginMeters;
    if (maximumMeters <= minimumMeters) return null;
    const betaGamma = momentumMagnitudeGeV / parent.mass;
    const properMinimum = minimumMeters / betaGamma;
    const properMaximum = maximumMeters / betaGamma;
    const properDistance = sampleConditionedExponential(
      parent.cTau * CONFIG.lifetimeScale,
      properMinimum,
      properMaximum,
    );
    const labDistanceMeters = properDistance * betaGamma;
    return {
      x: start.x + Math.cos(angle) * labDistanceMeters * CONFIG.pixelsPerMeter,
      y: start.y + Math.sin(angle) * labDistanceMeters * CONFIG.pixelsPerMeter,
      labDistanceMeters,
    };
  }

  function makeTrack(transport, delay = 0, endIndex = transport.points.length - 1) {
    const track = {
      particle: transport.particle,
      points: transport.points.slice(0, endIndex + 1),
      times: transport.times.slice(0, endIndex + 1),
      states: transport.states.slice(0, endIndex + 1),
      delay,
      tailHold: CONFIG.tailHoldMilliseconds,
      tailFade: CONFIG.tailFadeMilliseconds,
      stopReason: endIndex < transport.points.length - 1 ? "decayed" : transport.stopReason,
    };
    track.bubbles = physics.sampleBubbles(track, CONFIG);
    return track;
  }

  function trackEndTime(track) {
    return track.endTime ?? track.delay + track.times[track.times.length - 1];
  }

  function particleLatex(particle) {
    const names = {
      "π+": "\\pi^+",
      "π−": "\\pi^-",
      "π0": "\\pi^0",
      "μ+": "\\mu^+",
      "μ−": "\\mu^-",
      "e+": "e^+",
      "e−": "e^-",
      "γ": "\\gamma",
      "ν": "\\nu",
      "p": "p",
      "p̄": "\\bar{p}",
      "n": "n",
      "n̄": "\\bar{n}",
      "K⁰S": "K^0_{S}",
      "Λ": "\\Lambda",
      "Λ̄": "\\bar{\\Lambda}",
      "K+": "K^+",
      "K−": "K^-",
      "Ξ−": "\\Xi^-",
      "Ξ̄+": "\\bar{\\Xi}^{+}",
      "γ*": "\\gamma^*",
    };
    return names[particle.name] || `\\mathrm{${particle.name}}`;
  }

  function renderLatex(source, target, fallback = source) {
    if (typeof window.katex === "undefined") {
      target.textContent = fallback;
      return;
    }
    window.katex.render(source, target, {
      displayMode: false,
      throwOnError: false,
      strict: false,
    });
  }

  function particleEntry(particle, role) {
    return {
      name: particle.name,
      latex: particleLatex(particle),
      role,
      charge: particle.charge,
      visible: particle.charge !== 0,
    };
  }

  function makeEventRecord(kind, label, reaction, reactionLatex, explanation, particles, tracks) {
    return { kind, label, reaction, reactionLatex, explanation, particles, tracks };
  }

  function makeSingleTrack(startOverride = null) {
    const particle = PARTICLES[chooseWeighted([
      { weight: 4, value: "pi+" },
      { weight: 4, value: "pi-" },
      { weight: 1, value: "p" },
      { weight: 1, value: "pbar" },
      { weight: 1, value: "mu+" },
      { weight: 1, value: "mu-" },
    ]).value];
    const start = startOverride ? clickStart(startOverride) : beamStart();
    const momentum = momentumFromMagnitude(singleTrackMomentum(), start.angle);
    return makeEventRecord(
      "track",
      "Single track",
      particle.name,
      particleLatex(particle),
      "A single charged particle bends in the magnetic field. The bend direction depends on its charge, and tighter curvature indicates lower momentum.",
      [particleEntry(particle, "charged particle")],
      [makeTrack(transportCharged(particle, start, momentum))],
    );
  }

  function makeDecayEvent(startOverride = null) {
    const parent = PARTICLES[chooseWeighted([
      { weight: 3.2, value: "KS" },
      { weight: 1.5, value: "Lambda" },
      { weight: 1.5, value: "LambdaBar" },
      { weight: 1, value: "K+" },
      { weight: 1, value: "K-" },
    ]).value];
    const start = startOverride ? clickStart(startOverride) : inwardStart();
    const parentMomentumMagnitude =
      parent.charge === 0 ? logUniform(0.22, 1.25) : logUniform(0.25, 1.05);
    let vertex;
    let parentMomentum;
    let parentTrack = null;
    let daughterDelay = 0;

    if (parent.charge === 0) {
      vertex = selectNeutralDecay(parent, start, start.angle, parentMomentumMagnitude);
      if (!vertex) return makeSingleTrack(startOverride);
      parentMomentum = momentumFromMagnitude(parentMomentumMagnitude, start.angle);
    } else {
      const initialMomentum = momentumFromMagnitude(parentMomentumMagnitude, start.angle);
      const parentTransport = transportCharged(parent, start, initialMomentum);
      const decayIndex = selectChargedDecayState(parent, parentTransport);
      if (decayIndex === null) return makeSingleTrack(startOverride);
      const state = parentTransport.states[decayIndex];
      vertex = { x: state.x, y: state.y };
      parentMomentum = { px: state.px, py: state.py };
      parentTrack = makeTrack(parentTransport, 0, decayIndex);
      daughterDelay = trackEndTime(parentTrack);
    }

    // Condition on an observable chamber topology. The relative branching
    // fractions among channels with at least one charged daughter are kept.
    const channel = chooseWeighted(
      parent.channels.filter((candidate) =>
        candidate.daughters.some((name) => PARTICLES[name].charge !== 0),
      ),
    );
    const daughters = channel.daughters.map((name) => PARTICLES[name]);
    const daughterVectors =
      daughters.length === 2
        ? twoBodyDecay(parent, parentMomentum, daughters[0], daughters[1])
        : threeBodyDecay(
            parent,
            parentMomentum,
            daughters[0],
            daughters[1],
            daughters[2],
          );
    const tracks = parentTrack ? [parentTrack] : [];
    for (let index = 0; index < daughters.length; index += 1) {
      const daughter = daughters[index];
      if (daughter.charge === 0) continue;
      const vector = daughterVectors[index];
      const daughterTransport = transportCharged(daughter, vertex, {
        px: vector.px,
        py: vector.py,
      });
      tracks.push(makeTrack(daughterTransport, daughterDelay));
    }
    return makeEventRecord(
      "decay",
      "Single decay",
      `${parent.name} → ${daughters.map((particle) => particle.name).join(" + ")}`,
      `${particleLatex(parent)} \\to ${daughters.map(particleLatex).join(" + ")}`,
      parent.charge === 0
        ? "The neutral parent leaves no track. Its decay becomes visible where the charged decay products begin their tracks."
        : "The charged parent leaves a track up to its decay point, where the charged decay products begin their tracks.",
      [particleEntry(parent, "parent"), ...daughters.map((particle) => particleEntry(particle, "daughter"))],
      tracks,
    );
  }

  function makeCascadeEvent(startOverride = null) {
    const parent = PARTICLES[Math.random() < 0.5 ? "Xi-" : "XiBar"];

    for (let attempt = 0; attempt < 10; attempt += 1) {
      const start = startOverride ? clickStart(startOverride) : inwardStart();
      const initialMomentum = momentumFromMagnitude(logUniform(0.32, 1.15), start.angle);
      const parentTransport = transportCharged(parent, start, initialMomentum);
      const cascadeIndex = selectChargedDecayState(parent, parentTransport);
      if (cascadeIndex === null) continue;

      const cascadeState = parentTransport.states[cascadeIndex];
      const cascadeVertex = { x: cascadeState.x, y: cascadeState.y };
      const cascadeMomentum = { px: cascadeState.px, py: cascadeState.py };
      const parentTrack = makeTrack(parentTransport, 0, cascadeIndex);
      const cascadeDelay = trackEndTime(parentTrack);
      const cascadeChannel = parent.channels[0];
      const lambda = PARTICLES[cascadeChannel.daughters[0]];
      const bachelorPion = PARTICLES[cascadeChannel.daughters[1]];
      const [lambdaVector, pionVector] = twoBodyDecay(
        parent,
        cascadeMomentum,
        lambda,
        bachelorPion,
      );

      const lambdaMomentum = Math.hypot(lambdaVector.px, lambdaVector.py);
      const lambdaAngle = Math.atan2(lambdaVector.py, lambdaVector.px);
      const lambdaVertex = selectNeutralDecay(
        lambda,
        cascadeVertex,
        lambdaAngle,
        lambdaMomentum,
      );
      if (!lambdaVertex) continue;

      const lambdaBeta = lambdaMomentum / Math.hypot(lambda.mass, lambdaMomentum);
      const lambdaDelay =
        cascadeDelay +
        (CONFIG.visualMillisecondsPerMeterAtC * lambdaVertex.labDistanceMeters) /
          Math.max(lambdaBeta, 0.03);
      const tracks = [
        parentTrack,
        makeTrack(
          transportCharged(bachelorPion, cascadeVertex, {
            px: pionVector.px,
            py: pionVector.py,
          }),
          cascadeDelay,
        ),
      ];

      const lambdaChannel = chooseWeighted(
        lambda.channels.filter((candidate) =>
          candidate.daughters.some((name) => PARTICLES[name].charge !== 0),
        ),
      );
      const lambdaDaughters = lambdaChannel.daughters.map((name) => PARTICLES[name]);
      const lambdaVectors =
        lambdaDaughters.length === 2
          ? twoBodyDecay(
              lambda,
              { px: lambdaVector.px, py: lambdaVector.py },
              lambdaDaughters[0],
              lambdaDaughters[1],
            )
          : threeBodyDecay(
              lambda,
              { px: lambdaVector.px, py: lambdaVector.py },
              lambdaDaughters[0],
              lambdaDaughters[1],
              lambdaDaughters[2],
            );
      for (let index = 0; index < lambdaDaughters.length; index += 1) {
        const daughter = lambdaDaughters[index];
        if (daughter.charge === 0) continue;
        tracks.push(
          makeTrack(
            transportCharged(daughter, lambdaVertex, {
              px: lambdaVectors[index].px,
              py: lambdaVectors[index].py,
            }),
            lambdaDelay,
          ),
        );
      }
      const finalDaughters = lambdaDaughters.map((particle) => particleEntry(particle, "daughter"));
      return makeEventRecord(
        "cascade",
        "Ξ cascade",
        `${parent.name} → ${lambda.name} + ${bachelorPion.name}; ${lambda.name} → ${lambdaDaughters.map((particle) => particle.name).join(" + ")}`,
        `${particleLatex(parent)} \\to ${particleLatex(lambda)} + ${particleLatex(bachelorPion)},\\quad ${particleLatex(lambda)} \\to ${lambdaDaughters.map(particleLatex).join(" + ")}`,
        "A charged Ξ decays into a pion and a neutral Λ. The Λ leaves no track before decaying again, so the two decay points are separated by a gap.",
        [
          particleEntry(parent, "parent"),
          particleEntry(lambda, "intermediate"),
          particleEntry(bachelorPion, "daughter"),
          ...finalDaughters,
        ],
        tracks,
      );
    }

    return makeDecayEvent(startOverride);
  }

  function makeDalitzEvent(vertexOverride = null) {
    const parent = PARTICLES.pi0;
    const electron = PARTICLES["e-"];
    const positron = PARTICLES["e+"];
    const gamma = PARTICLES.gamma;
    const vertex = vertexOverride
      ? {
          x: Math.max(0, Math.min(width, vertexOverride.x)),
          y: Math.max(0, Math.min(height, vertexOverride.y)),
        }
      : {
          x: randomBetween(width * 0.12, width * 0.88),
          y: randomBetween(height * 0.12, height * 0.88),
        };
    const parentMomentum = momentumFromMagnitude(
      logUniform(0.035, 0.28),
      (vertexOverride?.angle ?? beamAngle) + randomBetween(-0.8, 0.8),
    );

    // π0 -> γ γ*, followed by γ* -> e+ e−. This sequential construction is
    // an exact Dalitz-decay phase-space point and naturally produces the very
    // low-momentum electron curls characteristic of chamber photographs.
    const minimumPairMass = 2.02 * electron.mass;
    const virtualMass = logUniform(minimumPairMass, parent.mass * 0.72);
    const virtualPhoton = { name: "γ*", mass: virtualMass, charge: 0, cTau: 0 };
    const [virtualVector] = twoBodyDecay(
      parent,
      parentMomentum,
      virtualPhoton,
      gamma,
    );
    const [electronVector, positronVector] = twoBodyDecay(
      virtualPhoton,
      { px: virtualVector.px, py: virtualVector.py },
      electron,
      positron,
    );
    const tracks = [
      makeTrack(
        transportCharged(electron, vertex, {
          px: electronVector.px,
          py: electronVector.py,
        }),
      ),
      makeTrack(
        transportCharged(positron, vertex, {
          px: positronVector.px,
          py: positronVector.py,
        }),
      ),
    ];
    return makeEventRecord(
      "dalitz",
      "Dalitz decay",
      "π0 → γ + γ* → γ + e+ + e−",
      "\\pi^0 \\to \\gamma\\gamma^*,\\quad \\gamma^* \\to e^+e^-",
      "The π⁰ decays into a photon and an electron–positron pair. The photon leaves no track; the pair produces two tracks that curve in opposite directions from the same decay point.",
      [
        particleEntry(parent, "parent"),
        particleEntry(gamma, "daughter"),
        { name: "γ*", latex: "\\gamma^*", role: "intermediate", charge: 0, visible: false },
        particleEntry(positron, "daughter"),
        particleEntry(electron, "daughter"),
      ],
      tracks,
    );
  }

  function addEvent(now, eventRecord, ambient = false, layer = null) {
    const { tracks } = eventRecord;
    if (tracks.length === 0 || events.length >= 100) return false;
    if (playground) {
      if (tracks.length > TRACK_BUDGET) return false;
      let retained = events.reduce((count, event) => count + event.tracks.length, 0);
      while (events.length && retained + tracks.length > TRACK_BUDGET) {
        const index = events.findIndex((event) => event.record.kind !== "beam");
        if (index < 0) return false;
        retained -= events[index].tracks.length;
        removeEvent(index);
      }
    }
    const lifetime = Math.max(
      ...tracks.map((track) => trackEndTime(track) + track.tailHold + track.tailFade),
    );
    if (!layer) {
      layer = window.BubbleRenderer.createLayer(width, height, pixelRatio, { photograph: settings.mode === "photograph" });
      for (const track of tracks) layer.addTrack(track);
      for (const step of layer.prepare()) { /* small foreground events */ }
    }
    for (const track of tracks) {
      track.endTime = trackEndTime(track);
      // Transport samples are no longer needed once the display layer is built.
      delete track.points;
      delete track.states;
      delete track.times;
      delete track.bubbles;
    }
    events.push({ start: now, lifetime, tracks, layer, record: eventRecord, ambient });
    sceneRevision += 1;
    return true;
  }

  function makeRandomEvent(vertex = null) {
    const choice = Math.random();
    return choice < 0.35
        ? makeSingleTrack(vertex)
        : choice < 0.7
          ? makeDecayEvent(vertex)
          : choice < 0.88
            ? makeCascadeEvent(vertex)
            : makeDalitzEvent(vertex);
  }

  function makeBackgroundEvent() {
    const particle = PARTICLES[Math.random() < 0.5 ? "mu+" : "mu-"];
    const start = { x: randomBetween(width * 0.1, width * 0.9), y: -18,
      angle: Math.PI / 2 + randomBetween(-0.65, 0.65) };
    return { kind: "background", tracks: [makeTrack(transportCharged(particle, start,
      momentumFromMagnitude(randomBetween(2.5, 7), start.angle)))] };
  }

  // Posted-message tasks yield to input/rendering without the minimum delay
  // imposed on nested timers. Hundreds of small cache batches must not each
  // incur a timer-clamp wait before the completed beam can be displayed.
  const preparationQueue = [];
  const preparationChannel = new MessageChannel();
  preparationChannel.port1.onmessage = () => preparationQueue.shift()?.();
  const yieldToControls = () => new Promise((resolve) => {
    preparationQueue.push(resolve);
    preparationChannel.port2.postMessage(null);
  });

  async function prepareLayer(layer, job) {
    let batch = 0;
    let batchStarted = performance.now();
    for (const step of layer.prepare()) {
      if (job.cancelled) return false;
      if (++batch >= 4 || performance.now() - batchStarted >= 4) {
        await yieldToControls();
        batch = 0;
        batchStarted = performance.now();
      }
    }
    return !job.cancelled;
  }

  function exposureEventCount() {
    // Poisson occupancy: an average of N events does not force every plate to
    // contain exactly N. This is an illustrative rate, not a measured flux.
    const threshold = Math.exp(-settings.activity);
    let product = 1;
    let count = 0;
    do { product *= Math.random(); count += 1; } while (product > threshold);
    return count - 1;
  }

  function* materializeProduct(product, vertex, delay, vertices, role = "secondary") {
    const id = product.id;
    if (id === "gamma") return; // its energy remains in the invisible event record
    const particle = id === "K0" ? PARTICLES.KS : PARTICLES[id];
    const momentum = { px: product.px, py: product.py };
    let decayVertex = vertex;
    let decayMomentum = momentum;
    let decayDelay = delay;
    if (particle.charge !== 0) {
      const transported = transportCharged(particle, vertex, momentum);
      const decayIndex = id === "Xi-" ? selectChargedDecayState(particle, transported) : null;
      const track = makeTrack(transported, delay, decayIndex ?? transported.points.length - 1);
      track.role = role;
      yield track;
      if (decayIndex === null) return;
      const state = transported.states[decayIndex];
      decayVertex = { x: state.x, y: state.y };
      decayMomentum = { px: state.px, py: state.py };
      decayDelay = trackEndTime(track);
    } else if (id !== "pi0") {
      const magnitude = Math.hypot(product.px, product.py);
      const flight = selectNeutralDecay(particle, vertex, Math.atan2(product.py, product.px), magnitude);
      if (!flight) return; // a neutral product can leave the camera view
      decayVertex = { x: flight.x, y: flight.y };
      decayDelay += CONFIG.visualMillisecondsPerMeterAtC * flight.labDistanceMeters
        / Math.max(0.03, magnitude / product.energy);
    }
    const decay = interactions.decay(id, decayMomentum);
    if (!decay) return;
    vertices.push({ type: "decay", vertex: decayVertex, time: decayDelay,
      initial: [decay.parent], products: decay.products });
    for (const child of decay.products) yield* materializeProduct(child, decayVertex, decayDelay, vertices);
  }

  function beamPresent() {
    return events.some((event) => event.record.kind === "beam")
      || [...beamJobs].some((job) => !job.cancelled);
  }

  async function spawnBeam() {
    if (beamPresent()) return 0;
    const retained = events.filter((event) => event.record.kind === "beam")
      .reduce((count, event) => count + event.tracks.length, 0);
    const pending = [...beamJobs].filter((job) => !job.cancelled)
      .reduce((count, job) => count + job.count, 0);
    const count = Math.min(settings.beamDensity, MAXIMUM_BEAM_TRACKS - retained - pending);
    if (count <= 0 || events.length + beamJobs.size >= 100) return 0;
    const job = { count, cancelled: false };
    beamJobs.add(job);
    const button = document.querySelector(".playground-beam");
    button?.setAttribute("aria-busy", "true");
    const layer = window.BubbleRenderer.createLayer(width, height, pixelRatio, { photograph: settings.mode === "photograph" });
    if (settings.randomBeamAngle) {
      beamAngle = -Math.floor(Math.random() * 360) * Math.PI / 180;
      if (playground) updateBeamDirectionControl();
    }
    const direction = beamAngle;
    const tracks = [];
    const vertices = [];
    const selectedKind = playground ? selectedEventKind : "mixed";
    const additionalTracks = { track: 2, decay: 4, cascade: 7, dalitz: 4, mixed: 7 }[selectedKind];
    // One reaction per primary, with room for all associated products and for
    // manual clicks. Never truncate the products of an accepted interaction.
    const reactionCount = settings.automatic ? Math.min(exposureEventCount(), count,
      Math.max(0, Math.floor((80 - count) / additionalTracks))) : 0;
    const shuffled = Array.from({ length: count }, (_, index) => index);
    for (let index = shuffled.length - 1; index > 0; index -= 1) {
      const other = Math.floor(Math.random() * (index + 1));
      [shuffled[index], shuffled[other]] = [shuffled[other], shuffled[index]];
    }
    const reacting = new Set(shuffled.slice(0, reactionCount));
    let batchStarted = performance.now();
    let batchTracks = 0;
    let attached = false;
    try {
      // Yield even before the first pair, so clicks never synchronously transport
      // a whole burst. Geometry changes and Clear cancel outstanding jobs.
      await yieldToControls();
      for (let index = 0; index < count; index += 1) {
        if (job.cancelled) return null;
        const angle = direction + randomBetween(-0.004, 0.004);
        const offset = Math.max(-0.2, Math.min(0.2, physics.normal() * 0.045)) * Math.min(width, height);
        const start = physics.beamEntry(width, height, angle, offset);
        // Represent longitudinal bunch phase as an entry-time offset. Keep
        // the same entry port/direction, without extra bending in an invented
        // upstream stretch of magnet or liquid.
        const along = Math.min(520, Math.abs(physics.normal()) * 180);
        const magnitude = randomBetween(5, 9);
        const beta = magnitude / Math.hypot(PARTICLES["pi-"].mass, magnitude);
        const entryDelay = CONFIG.visualMillisecondsPerMeterAtC * along / CONFIG.pixelsPerMeter / beta;
        const transported = transportCharged(PARTICLES["pi-"], start,
          momentumFromMagnitude(magnitude, angle));
        let indexAtVertex = null;
        let production = null;
        if (reacting.has(index)) {
          const candidates = transported.states.map((state, position) => ({ state, position }))
            .filter(({ state }) => isOnScreen(state, -32));
          if (candidates.length > 8) {
            const selected = candidates[Math.floor(candidates.length * randomBetween(0.15, 0.7))];
            indexAtVertex = selected.position;
            production = interactions.produce(selectedKind, { px: selected.state.px, py: selected.state.py });
          }
        }
        const incoming = makeTrack(transported, entryDelay,
          production ? indexAtVertex : transported.points.length - 1);
        incoming.role = "beam-incoming";
        let generated = [incoming];
        if (production) {
          const vertex = transported.points[indexAtVertex];
          const time = entryDelay + transported.times[indexAtVertex];
          incoming.stopReason = "interaction";
          incoming.interaction = { ...production, vertex, time };
          vertices.push({ type: "production", ...incoming.interaction });
          generated = (function* () {
            yield incoming;
            for (const product of production.products) {
              yield* materializeProduct(product, vertex, time, vertices, product.id === "p" ? "recoil" : "secondary");
            }
          })();
        }
        for (const track of generated) {
          if (job.cancelled) return null;
          tracks.push(track);
          layer.addTrack(track);
          if (++batchTracks >= 2 || performance.now() - batchStarted >= 4) {
            await yieldToControls();
            batchTracks = 0;
            batchStarted = performance.now();
          }
        }
      }
      if (!await prepareLayer(layer, job)) return null;
      if (events.length >= 100) return 0;
      const startTime = settings.mode === "photograph" ? exposureStarted : simulationTime;
      attached = addEvent(startTime, { kind: "beam", primaryCount: count, tracks, vertices }, false, layer);
      draw(simulationTime);
      return count;
    } finally {
      if (!attached) layer.dispose();
      beamJobs.delete(job);
      if (beamJobs.size === 0) button?.removeAttribute("aria-busy");
    }
  }

  function spawnBackground(now) {
    const ambientCount = events.filter((event) => event.ambient).length;
    if (settings.backgroundTracks <= 0 || events.length >= 100 || ambientCount >= 4) return;
    addEvent(now, makeBackgroundEvent(), true);
  }

  function buildChamberSpecks() {
    sceneRevision += 1;
    specks = [];
    const count = Math.round(width * height / 4500 * settings.specks);
    // Most spurious nucleation is near the edge; a few fixed local sites add
    // spatial correlation. This is an optical approximation, not a boiling rate.
    const sites = Array.from({ length: 3 }, () => ({
      x: Math.random() < 0.5 ? randomBetween(0, 30) : width - randomBetween(0, 30),
      y: randomBetween(0, height),
    }));
    for (let index = 0; index < count; index += 1) {
      const clustered = index % 3 === 0;
      const site = sites[Math.floor(index / 3) % sites.length];
      specks.push({
        x: clustered ? site.x + physics.normal() * 12 : randomBetween(0, width),
        y: clustered ? site.y + physics.normal() * 20 : randomBetween(0, height),
        radius: randomBetween(0.48, clustered ? 1.45 : 1.0),
        strength: randomBetween(0.45, 0.85),
      });
    }
  }

  function beginExposure(now, includeBeam = true) {
    clearEvents();
    exposureStarted = now;
    exposureInitialized = true;
    buildChamberSpecks();
    if (includeBeam && settings.beamAutomatic) spawnBeam();
    if (settings.backgroundTracks > 0) spawnBackground(now);

  }

  function preparePhotoEvent(includeBeam = true) {
    if (settings.mode === "photograph") beginExposure(simulationTime, includeBeam);
  }

  function replayTiming() {
    if (!playground) return { hold: CONFIG.tailHoldMilliseconds, fade: CONFIG.tailFadeMilliseconds };
    // Budget visible tracks rather than making the visitor balance rate and
    // lifetime. Include beam occupancy and a small allowance for flight time.
    const arrivals = (settings.automatic && settings.beamAutomatic ? settings.activity * 2.2 / (BEAM_INTERVAL / 1000) : 0)
      + (settings.beamAutomatic ? settings.beamDensity / (BEAM_INTERVAL / 1000) : 0)
      + settings.backgroundTracks * 0.175;
    const seconds = Math.max(6, Math.min(26, 40 / Math.max(0.1, arrivals)));
    const fade = Math.min(2000, seconds * 150);
    return { hold: Math.max(2000, seconds * 1000 - fade - 2000), fade };
  }

  function chargeLabel(charge) {
    if (charge > 0) return `charge +${charge}`;
    if (charge < 0) return `charge ${charge}`;
    return "neutral";
  }

  function renderEventReadout(eventRecord) {
    const heading = document.createElement("p");
    heading.className = "bubble-chamber-event-readout__label";
    heading.textContent = eventRecord.label;

    const reaction = document.createElement("h3");
    renderLatex(eventRecord.reactionLatex, reaction, eventRecord.reaction);

    const explanation = document.createElement("p");
    explanation.textContent = eventRecord.explanation;

    const particleList = document.createElement("ul");
    for (const particle of eventRecord.particles) {
      const item = document.createElement("li");
      const name = document.createElement("strong");
      const details = document.createElement("span");
      renderLatex(particle.latex, name, particle.name);
      details.textContent = `${particle.role} · ${chargeLabel(particle.charge)} · ${particle.visible ? "visible track" : "no track"}`;
      item.className = particle.visible ? "is-visible" : "is-invisible";
      item.append(name, details);
      particleList.append(item);
    }

    eventReadout.replaceChildren(heading, reaction, explanation, particleList);
  }

  function updateEventKindControls() {
    for (const button of explorer.querySelectorAll("[data-event-kind]")) {
      const selected = button.dataset.eventKind === selectedEventKind;
      button.classList.toggle("is-selected", selected);
      button.setAttribute("aria-pressed", String(selected));
    }
  }

  function makeSelectedEvent(kind, vertexOverride = null) {
    if (kind === "mixed") return makeRandomEvent(vertexOverride);
    const factories = {
      decay: () => makeDecayEvent(vertexOverride),
      cascade: () => makeCascadeEvent(vertexOverride),
      dalitz: () => makeDalitzEvent(vertexOverride),
      track: () => makeSingleTrack(vertexOverride),
    };
    const factory = factories[kind];
    if (!factory) return null;

    let eventRecord = factory();
    for (let attempt = 0; attempt < 6 && eventRecord.kind !== kind; attempt += 1) {
      eventRecord = factory();
    }
    return eventRecord.kind === kind ? eventRecord : null;
  }

  function generateExploredEvent(kind) {
    selectedEventKind = kind;
    updateEventKindControls();
    if (paused) {
      showStatus("Resume the chamber to generate an event.");
      return;
    }

    const eventRecord = makeSelectedEvent(kind);
    if (!eventRecord) {
      showStatus("Could not generate this event. Try again.");
      return;
    }
    preparePhotoEvent();
    if (!playground) clearEvents();
    if (!addEvent(simulationTime, eventRecord)) {
      showStatus("No visible tracks were generated. Try again.");
      return;
    }
    renderEventReadout(eventRecord);
  }

  function handleClick(event) {
    if (event.defaultPrevented || event.button !== 0) return;
    if (
      event.target.closest(
        ".bubble-chamber-controls, .playground-panel, .laddu-spectrum, a, button, input, select, label, summary, video, audio",
      )
    ) return;
    if (window.getSelection()?.toString()) return;

    if (paused) {
      showStatus("Resume the chamber to generate an event.");
      return;
    }

    if (playground && events.length >= 100) {
      showStatus("Chamber full. Clear tracks to create more events.");
      return;
    }
    if (!playground) {
      while (events.length >= CONFIG.maximumConcurrentEvents) removeEvent(0);
    }
    const eventRecord = makeSelectedEvent(selectedEventKind, {
      x: event.clientX,
      y: event.clientY,
    });
    if (eventRecord) preparePhotoEvent();
    if (!eventRecord || !addEvent(simulationTime, eventRecord)) {
      showStatus("Could not generate this event. Try again.");
      return;
    }
    if (explorer.open) renderEventReadout(eventRecord);
  }

  function showStatus(message) {
    window.clearTimeout(statusTimeout);
    status.textContent = message;
    status.classList.add("visible");
    statusTimeout = window.setTimeout(() => status.classList.remove("visible"), 2400);
  }

  function updatePauseControl() {
    const label = paused ? "Resume bubble chamber" : "Pause bubble chamber";
    pauseButton.setAttribute("aria-label", label);
    pauseButton.setAttribute("title", label);
    pauseButton.setAttribute("aria-pressed", String(paused));
    pauseButton.querySelector("span").textContent = paused ? "▶" : "Ⅱ";
    controls.classList.toggle("paused", paused);
    if (playground) {
      const button = document.querySelector("#playground-pause");
      button.textContent = paused ? "Resume" : "Pause";
      button.setAttribute("aria-pressed", String(paused));
    }
  }

  function setPaused(nextPaused) {
    if (paused === nextPaused) return;
    paused = nextPaused;
    if (!playground) storePausedPreference(paused);
    previousFrame = 0;
    if (paused) {
      window.cancelAnimationFrame(animationFrame);
      animationFrame = 0;
      draw(simulationTime);
      showStatus("Bubble chamber paused");
    } else {
      if (!document.hidden && !animationFrame) {
        animationFrame = window.requestAnimationFrame(animate);
      }
      showStatus("Bubble chamber resumed");
    }

    updatePauseControl();
  }

  function draw(now) {
    const photo = settings.mode === "photograph";
    const exposure = { scale: 1 };
    const timing = replayTiming();
    const color = getComputedStyle(canvas).color;
    const developed = photo && now - exposureStarted >= EXPOSURE.growth
      && events.every((event) => now - event.start >= EXPOSURE.growth);
    const key = developed ? `${sceneRevision}:${events.length}:${color}` : null;
    // A held photograph does not change. Avoid repainting thousands of bubbles
    // until the scene, appearance, or exposure phase actually changes.
    if (key !== null && key === heldPhotoKey) return;
    heldPhotoKey = key;
    context.clearRect(0, 0, width, height);
    context.fillStyle = "black";
    const speckGrowth = photo ? Math.max(0, Math.min(1, (now - exposureStarted) / EXPOSURE.growth)) : 1;
    for (const bubble of specks) {
      context.globalAlpha = bubble.strength * speckGrowth;
      context.beginPath();
      context.arc(bubble.x, bubble.y, bubble.radius * (0.2 + 0.8 * Math.sqrt(speckGrowth)), 0, Math.PI * 2);
      context.fill();
    }
    for (let index = events.length - 1; index >= 0; index -= 1) {
      const event = events[index];
      const age = now - event.start;
      const lifetime = playground
        ? Math.max(...event.tracks.map((track) => trackEndTime(track) + timing.hold + timing.fade))
        : event.lifetime;
      if (!photo && age >= lifetime) {
        removeEvent(index);
        continue;
      }
      const opacity = (event.ambient ? 0.48 : 1) * exposure.scale;
      if (opacity > 0) event.layer.draw(context, { age, opacity, photo,
        growth: Math.max(0, Math.min(1, age / EXPOSURE.growth)), scale: 1,
        hold: timing.hold, fade: timing.fade, keep: false });
    }
    context.globalAlpha = 1;
    context.globalCompositeOperation = "source-in";
    context.fillStyle = color;
    context.fillRect(0, 0, width, height);
    context.globalCompositeOperation = "source-over";
  }

  function animate(frameTime) {
    if (previousFrame) simulationTime += (frameTime - previousFrame) * settings.speed;
    previousFrame = frameTime;
    const now = simulationTime;
    if (settings.mode === "photograph") {
      const period = EXPOSURE.growth + EXPOSURE.hold;
      if (!exposureInitialized || now - exposureStarted >= period) beginExposure(now);
    } else {
      if (settings.beamAutomatic && now >= nextBeamAt && !beamPresent()) {
        nextBeamAt = now + 300;
        spawnBeam();
      }
      if (now >= nextBackgroundAt) {
        nextBackgroundAt = now + 2000;
        if (Math.random() < settings.backgroundTracks * 0.35) spawnBackground(now);
      }
    }
    draw(now);
    animationFrame = window.requestAnimationFrame(animate);
  }

  function handleVisibility() {
    if (document.hidden) {
      window.cancelAnimationFrame(animationFrame);
      animationFrame = 0;
      return;
    }
    if (!paused && !animationFrame) {
      previousFrame = 0;
      if (!playground) clearEvents();
      animationFrame = window.requestAnimationFrame(animate);
    }
  }

  function updateBeamDirectionControl() {
    const degrees = settings.randomBeamAngle
      ? (Math.round(-beamAngle * 180 / Math.PI) + 360) % 360 : manualBeamDegrees;
    const direction = document.querySelector("#playground-beam-direction");
    direction.value = degrees;
    direction.disabled = settings.randomBeamAngle;
    document.querySelector("#playground-beam-direction-value").textContent = `${degrees}°`;
    document.querySelector("#playground-beam-arrow").style.transform = `rotate(${-degrees}deg)`;
  }

  function updatePlaygroundControls() {
    const photo = settings.mode === "photograph";
    const activity = settings.activity;
    document.querySelector("#playground-event-rate-value").textContent = activity === 0 ? "Off" : `${activity} / beam`;
    const rate = document.querySelector("#playground-event-rate");
    rate.max = "10";
    rate.value = activity;
    rate.setAttribute("aria-valuetext", `${activity} reactions per beam passage on average`);
    document.querySelector("#playground-motion").hidden = photo;
    document.querySelector("#playground-clear").textContent = photo ? "Next photo" : "Clear";
    const speed = settings.replaySpeed;
    document.querySelector("#playground-speed").value = speed;
    document.querySelector("#playground-speed-value").textContent = `${speed}×`;
    document.querySelector("#playground-speed").setAttribute("aria-valuetext", `${speed} times normal speed`);
    document.querySelector("#playground-beam-density-value").textContent = settings.beamDensity ? `${settings.beamDensity} tracks` : "Off";
  }

  function initializePlayground() {
    selectedEventKind = "mixed";
    document.querySelector("#playground-topology").addEventListener("change", (event) => {
      selectedEventKind = event.target.value;
    });
    document.querySelector("#playground-beam-density").addEventListener("input", (event) => {
      settings.beamDensity = Number(event.target.value);
      settings.beamAutomatic = settings.beamDensity > 0;
      if (!settings.beamAutomatic) {
        cancelBeams();
        for (let index = events.length - 1; index >= 0; index -= 1) {
          if (events[index].record.kind === "beam") removeEvent(index);
        }
      }
      nextBeamAt = simulationTime + 300;
      updatePlaygroundControls();
      draw(simulationTime);
    });
    document.querySelector("#playground-view").addEventListener("change", (event) => {
      settings.mode = event.target.value;
      settings.speed = settings.mode === "photograph" ? 1 : settings.replaySpeed;
      settings.automatic = settings.activity > 0;
      clearEvents();
      exposureInitialized = false;
      if (settings.mode === "photograph") beginExposure(simulationTime - (paused ? EXPOSURE.growth : 0));
      updatePlaygroundControls();
      draw(simulationTime);
    });
    document.querySelector("#playground-random-angle").addEventListener("change", (event) => {
      settings.randomBeamAngle = event.target.checked;
      if (!settings.randomBeamAngle) beamAngle = -manualBeamDegrees * Math.PI / 180;
      updateBeamDirectionControl();
    });
    document.querySelector("#playground-beam-direction").addEventListener("input", (event) => {
      manualBeamDegrees = Number(event.target.value);
      beamAngle = -manualBeamDegrees * Math.PI / 180;
      updateBeamDirectionControl();
    });
    document.querySelector("#playground-speed").addEventListener("input", (event) => {
      settings.replaySpeed = Number(event.target.value);
      settings.speed = settings.replaySpeed;
      updatePlaygroundControls();
    });
    document.querySelector("#playground-event-rate").addEventListener("input", (event) => {
      const activity = Number(event.target.value);
      settings.activity = activity;
      settings.automatic = activity > 0;
      updatePlaygroundControls();
      draw(simulationTime);
    });
    document.querySelector("#playground-pause").addEventListener("click", () => setPaused(!paused));
    document.querySelector("#playground-clear").addEventListener("click", () => {
      if (settings.mode === "photograph") beginExposure(simulationTime - (paused ? EXPOSURE.growth : 0));
      else clearEvents();
      draw(simulationTime);
    });
    document.querySelector("#playground-export").addEventListener("click", exportChamber);
    updatePlaygroundControls();
  }

  function exportChamber() {
    draw(simulationTime);
    const output = document.createElement("canvas");
    output.width = canvas.width;
    output.height = canvas.height;
    const outputContext = output.getContext("2d");
    if (!outputContext) {
      showStatus("PNG export is unavailable in this browser.");
      return;
    }
    outputContext.fillStyle = getComputedStyle(document.body).backgroundColor;
    outputContext.fillRect(0, 0, output.width, output.height);
    outputContext.drawImage(canvas, 0, 0);
    output.toBlob((blob) => {
      if (!blob) {
        showStatus("PNG export failed. Please try again.");
        return;
      }
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "bubble-chamber.png";
      link.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
      showStatus("Chamber exported as PNG");
    }, "image/png");
  }

  window.addEventListener("resize", resize, { passive: true });
  pauseButton.addEventListener("click", () => setPaused(!paused));
  explorer.addEventListener("click", (event) => {
    const button = event.target.closest("[data-event-kind]");
    if (button) generateExploredEvent(button.dataset.eventKind);
  });
  explorer.addEventListener("toggle", () => {
    if (explorer.open) infoPanel.open = false;
  });
  infoPanel.addEventListener("toggle", () => {
    if (infoPanel.open) explorer.open = false;
  });
  document.addEventListener("click", (event) => {
    if (infoPanel.open && !infoPanel.contains(event.target)) {
      infoPanel.open = false;
    }
    if (explorer.open && !explorer.contains(event.target)) explorer.open = false;
  });
  document.addEventListener("click", handleClick);
  document.addEventListener("visibilitychange", handleVisibility);
  reducedMotion.addEventListener("change", () => {
    if (playground) setPaused(reducedMotion.matches);
    else window.location.reload();
  });
  for (const button of explorer.querySelectorAll("[data-event-kind]")) {
    const topology = button.querySelector(".bubble-chamber-topology");
    renderLatex(TOPOLOGY_LATEX[button.dataset.eventKind], topology, topology.textContent);
  }
  if (playground) initializePlayground();
  updateEventKindControls();
  resize();
  updatePauseControl();
  if (paused) draw(simulationTime);
  else animationFrame = window.requestAnimationFrame(animate);
})();
