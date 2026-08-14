(() => {
  "use strict";

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  if (reducedMotion.matches) return;

  // Visual handles. Particle masses, charges, and cτ values below remain in
  // physical units; these constants map the simulated chamber onto the page.
  const CONFIG = {
    magneticFieldTesla: 1.4,
    pixelsPerMeter: 360,
    stepPixels: 3.5,
    visualMillisecondsPerMeterAtC: 800,
    lifetimeScale: 8,
    minimumIonizationLossGeVPerMeter: 0.003,
    electronIonizationLossGeVPerMeter: 0.055,
    tailHoldMilliseconds: 9000,
    tailFadeMilliseconds: 4000,
    maximumTransportSteps: 14000,
    eventCheckIntervalMilliseconds: 1000,
    eventChancePerCheck: 0.10,
    maximumConcurrentEvents: 4,
  };
  const CURVATURE_GEV_PER_TESLA_METER = 0.299792458;

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
    n: { name: "n", mass: 0.939565421, charge: 0, cTau: 0.265 },
    nbar: { name: "n̄", mass: 0.939565421, charge: 0, cTau: 0.265 },
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

  const events = [];
  let width = 0;
  let height = 0;
  let pixelRatio = 1;
  let animationFrame = 0;
  let beamAngle = randomBetween(-0.18, 0.18);
  let nextEventCheckAt = performance.now() + CONFIG.eventCheckIntervalMilliseconds;

  function randomBetween(minimum, maximum) {
    return minimum + Math.random() * (maximum - minimum);
  }

  function logUniform(minimum, maximum) {
    return Math.exp(randomBetween(Math.log(minimum), Math.log(maximum)));
  }

  function throughTrackMomentum() {
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
    width = window.innerWidth;
    height = window.innerHeight;
    pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * pixelRatio);
    canvas.height = Math.round(height * pixelRatio);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    events.length = 0;
    beamAngle = randomBetween(-0.18, 0.18);
    nextEventCheckAt = performance.now() + CONFIG.eventCheckIntervalMilliseconds;
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
    const fromLeft = Math.random() < 0.75;
    if (fromLeft) {
      return {
        x: -18,
        y: randomBetween(height * 0.08, height * 0.92),
        angle: beamAngle + randomBetween(-0.035, 0.035),
      };
    }
    return {
      x: width + 18,
      y: randomBetween(height * 0.08, height * 0.92),
      angle: Math.PI + beamAngle + randomBetween(-0.035, 0.035),
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
    const points = [{ x: start.x, y: start.y }];
    const times = [0];
    const states = [];
    const stepMeters = CONFIG.stepPixels / CONFIG.pixelsPerMeter;
    let x = start.x;
    let y = start.y;
    let px = initialMomentum.px;
    let py = initialMomentum.py;
    let elapsed = 0;
    let properLength = 0;
    let pathLength = 0;
    let entered = isOnScreen(start);
    let exitedOnce = false;
    let returnedAfterExit = false;
    let stopReason = "transport-limit";

    states.push({ x, y, px, py, properLength, pathLength });
    for (let index = 1; index <= CONFIG.maximumTransportSteps; index += 1) {
      const momentum = Math.hypot(px, py);
      const totalEnergy = Math.hypot(particle.mass, momentum);
      const beta = momentum / totalEnergy;
      const betaGamma = momentum / particle.mass;
      if (totalEnergy - particle.mass < 0.0008 || momentum < 0.001) {
        stopReason = "ranged-out";
        break;
      }

      let angle = Math.atan2(py, px);
      const radiusMeters =
        momentum /
        (CURVATURE_GEV_PER_TESLA_METER * Math.abs(particle.charge) * CONFIG.magneticFieldTesla);
      angle += (particle.charge * stepMeters) / radiusMeters;
      x += Math.cos(angle) * CONFIG.stepPixels;
      y += Math.sin(angle) * CONFIG.stepPixels;
      pathLength += stepMeters;
      properLength += stepMeters / Math.max(betaGamma, 1e-6);
      elapsed += (CONFIG.visualMillisecondsPerMeterAtC * stepMeters) / Math.max(beta, 0.03);

      const baseStoppingPower =
        particle.mass < 0.01
          ? CONFIG.electronIonizationLossGeVPerMeter
          : CONFIG.minimumIonizationLossGeVPerMeter;
      const stoppingPower = baseStoppingPower / Math.max(beta * beta, 0.1);
      const nextEnergy = Math.max(particle.mass, totalEnergy - stoppingPower * stepMeters);
      const nextMomentum = Math.sqrt(Math.max(0, nextEnergy * nextEnergy - particle.mass * particle.mass));
      px = nextMomentum * Math.cos(angle);
      py = nextMomentum * Math.sin(angle);
      points.push({ x, y });
      times.push(elapsed);
      states.push({ x, y, px, py, properLength, pathLength });
      const onScreen = isOnScreen({ x, y });
      const nearScreen = isOnScreen({ x, y }, 26);
      entered ||= onScreen;
      if (exitedOnce && onScreen) returnedAfterExit = true;
      if (entered && !nearScreen) {
        const radiusPixels = radiusMeters * CONFIG.pixelsPerMeter;
        if (!exitedOnce) {
          exitedOnce = true;
          // Tight tracks remain in the simulated chamber so their next orbit
          // can cross the viewport again. Broad tracks are retired off-screen.
          if (radiusPixels > Math.min(width, height) * 0.7) {
            stopReason = "exited";
            break;
          }
        } else if (returnedAfterExit) {
          stopReason = "exited-after-return";
          break;
        }
      }
    }

    return { particle, points, times, states, stopReason };
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

  function addTexture(track) {
    const isElectron = track.particle.mass < 0.01;
    track.segmentStrengths = new Array(track.points.length).fill(1);
    track.lineWidth = isElectron ? 0.38 : 0.58;
    let gapRemaining = 0;
    let density = isElectron ? randomBetween(0.985, 0.997) : randomBetween(0.68, 0.9);
    for (let index = 1; index < track.points.length; index += 1) {
      if (gapRemaining > 0) {
        track.segmentStrengths[index] = 0;
        gapRemaining -= 1;
        continue;
      }
      if (Math.random() > density) {
        gapRemaining = isElectron ? 1 : Math.floor(randomBetween(1, 4));
        track.segmentStrengths[index] = 0;
        continue;
      }
      track.segmentStrengths[index] = isElectron
        ? randomBetween(0.78, 1)
        : randomBetween(0.48, 1);
      if (!isElectron && Math.random() < 0.025) density = randomBetween(0.62, 0.94);
    }
    return track;
  }

  function makeTrack(transport, delay = 0, endIndex = transport.points.length - 1) {
    const points = transport.points.slice(0, endIndex + 1);
    const times = transport.times.slice(0, endIndex + 1);
    return addTexture({
      particle: transport.particle,
      points,
      times,
      delay,
      tailHold: CONFIG.tailHoldMilliseconds,
      tailFade: CONFIG.tailFadeMilliseconds,
      stopReason: endIndex < transport.points.length - 1 ? "decayed" : transport.stopReason,
      segmentStrengths: [],
    });
  }

  function trackEndTime(track) {
    return track.delay + track.times[track.times.length - 1];
  }

  function makeThroughTrack() {
    const particle = PARTICLES[chooseWeighted([
      { weight: 4, value: "pi+" },
      { weight: 4, value: "pi-" },
      { weight: 1, value: "p" },
      { weight: 1, value: "pbar" },
      { weight: 1, value: "mu+" },
      { weight: 1, value: "mu-" },
    ]).value];
    const start = beamStart();
    const momentum = momentumFromMagnitude(throughTrackMomentum(), start.angle);
    return [makeTrack(transportCharged(particle, start, momentum))];
  }

  function makeDecayEvent() {
    const parent = PARTICLES[chooseWeighted([
      { weight: 3.2, value: "KS" },
      { weight: 1.5, value: "Lambda" },
      { weight: 1.5, value: "LambdaBar" },
      { weight: 1, value: "K+" },
      { weight: 1, value: "K-" },
    ]).value];
    const start = inwardStart();
    const parentMomentumMagnitude =
      parent.charge === 0 ? logUniform(0.22, 1.25) : logUniform(0.25, 1.05);
    let vertex;
    let parentMomentum;
    let parentTrack = null;
    let daughterDelay = 0;

    if (parent.charge === 0) {
      vertex = selectNeutralDecay(parent, start, start.angle, parentMomentumMagnitude);
      if (!vertex) return makeThroughTrack();
      parentMomentum = momentumFromMagnitude(parentMomentumMagnitude, start.angle);
    } else {
      const initialMomentum = momentumFromMagnitude(parentMomentumMagnitude, start.angle);
      const parentTransport = transportCharged(parent, start, initialMomentum);
      const decayIndex = selectChargedDecayState(parent, parentTransport);
      if (decayIndex === null) return makeThroughTrack();
      const state = parentTransport.states[decayIndex];
      vertex = { x: state.x, y: state.y };
      parentMomentum = { px: state.px, py: state.py };
      parentTrack = makeTrack(parentTransport, 0, decayIndex);
      daughterDelay = trackEndTime(parentTrack);
    }

    const channel = chooseWeighted(parent.channels);
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
    return tracks;
  }

  function makeCascadeEvent() {
    const parent = PARTICLES[Math.random() < 0.5 ? "Xi-" : "XiBar"];

    for (let attempt = 0; attempt < 10; attempt += 1) {
      const start = inwardStart();
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

      const lambdaChannel = chooseWeighted(lambda.channels);
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
      return tracks;
    }

    return makeDecayEvent();
  }

  function makeDalitzEvent() {
    const parent = PARTICLES.pi0;
    const electron = PARTICLES["e-"];
    const positron = PARTICLES["e+"];
    const gamma = PARTICLES.gamma;
    const vertex = {
      x: randomBetween(width * 0.12, width * 0.88),
      y: randomBetween(height * 0.12, height * 0.88),
    };
    const parentMomentum = momentumFromMagnitude(
      logUniform(0.035, 0.28),
      beamAngle + randomBetween(-0.8, 0.8),
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
    return [
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
  }

  function spawnEvent(now) {
    const choice = Math.random();
    const tracks =
      choice < 0.35
        ? makeThroughTrack()
        : choice < 0.7
          ? makeDecayEvent()
          : choice < 0.88
            ? makeCascadeEvent()
            : makeDalitzEvent();
    if (tracks.length === 0) return false;
    const lifetime = Math.max(
      ...tracks.map((track) => trackEndTime(track) + track.tailHold + track.tailFade),
    );
    events.push({ start: now, lifetime, tracks });
    return true;
  }

  function lowerBound(values, target) {
    let low = 0;
    let high = values.length;
    while (low < high) {
      const middle = (low + high) >> 1;
      if (values[middle] < target) low = middle + 1;
      else high = middle;
    }
    return low;
  }

  function upperBound(values, target) {
    let low = 0;
    let high = values.length;
    while (low < high) {
      const middle = (low + high) >> 1;
      if (values[middle] <= target) low = middle + 1;
      else high = middle;
    }
    return low;
  }

  function pointOpacity(eventAge, track, pointIndex) {
    const age = eventAge - track.delay - track.times[pointIndex];
    if (age < 0) return 0;
    if (age <= track.tailHold) return 1;
    return Math.max(0, 1 - (age - track.tailHold) / track.tailFade);
  }

  function drawTrack(track, eventAge) {
    const localAge = eventAge - track.delay;
    if (localAge < 0) return;
    const newestIndex = Math.min(track.points.length - 1, upperBound(track.times, localAge) - 1);
    const oldestIndex = Math.max(
      1,
      lowerBound(track.times, localAge - track.tailHold - track.tailFade),
    );
    context.lineWidth = track.lineWidth;
    for (let index = oldestIndex; index <= newestIndex; index += 1) {
      const strength = track.segmentStrengths[index];
      if (strength <= 0) continue;
      context.globalAlpha = pointOpacity(eventAge, track, index) * strength;
      context.beginPath();
      context.moveTo(track.points[index - 1].x, track.points[index - 1].y);
      context.lineTo(track.points[index].x, track.points[index].y);
      context.stroke();
    }
  }

  function draw(now) {
    context.clearRect(0, 0, width, height);
    const color = getComputedStyle(canvas).color;
    context.strokeStyle = color;
    context.fillStyle = color;
    context.lineCap = "round";
    for (let index = events.length - 1; index >= 0; index -= 1) {
      const event = events[index];
      const eventAge = now - event.start;
      if (eventAge >= event.lifetime) {
        events.splice(index, 1);
        continue;
      }
      for (const track of event.tracks) drawTrack(track, eventAge);
    }
    context.globalAlpha = 1;
  }

  function animate(now) {
    draw(now);

    if (now >= nextEventCheckAt) {
      const missedChecks = Math.floor(
        (now - nextEventCheckAt) / CONFIG.eventCheckIntervalMilliseconds,
      );
      nextEventCheckAt +=
        (missedChecks + 1) * CONFIG.eventCheckIntervalMilliseconds;
      const chanceAcrossChecks =
        1 - Math.pow(1 - CONFIG.eventChancePerCheck, missedChecks + 1);
      if (
        events.length < CONFIG.maximumConcurrentEvents &&
        Math.random() < chanceAcrossChecks
      ) {
        spawnEvent(now);
      }
    }
    animationFrame = window.requestAnimationFrame(animate);
  }

  function handleVisibility() {
    if (document.hidden) {
      window.cancelAnimationFrame(animationFrame);
      animationFrame = 0;
      return;
    }
    if (!animationFrame) {
      events.length = 0;
      nextEventCheckAt =
        performance.now() + CONFIG.eventCheckIntervalMilliseconds;
      animationFrame = window.requestAnimationFrame(animate);
    }
  }

  window.addEventListener("resize", resize, { passive: true });
  document.addEventListener("visibilitychange", handleVisibility);
  reducedMotion.addEventListener("change", () => window.location.reload());
  resize();
  animationFrame = window.requestAnimationFrame(animate);
})();
