/* Liquid-hydrogen transport and bubble sampling, shared by the browser and tests.
 * Physics units: GeV and metres (including cτ); animation times are slowed milliseconds.
 * Material and Sternheimer parameters:
 * https://pdg.lbl.gov/2017/AtomicNuclearProperties/HTML/liquid_hydrogen.html
 * https://pdg.lbl.gov/2017/AtomicNuclearProperties/MUE/muE_liquid_hydrogen.txt
 * Bethe / electron collision losses and Gaussian scattering core:
 * https://pdg.lbl.gov/2024/reviews/rpp2024-rev-passage-particles-matter.pdf
 * This omits hard scattering tails, straggling, nuclear interactions and showers.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.BubblePhysics = factory();
})(typeof globalThis === "object" ? globalThis : this, function () {
  "use strict";

  const HYDROGEN = Object.freeze({
    name: "Liquid hydrogen",
    densityGramsPerCubicCentimeter: 0.0708,
    radiationLengthMeters: 8.905,
    excitationEnergyGeV: 21.8e-9,
    zOverA: 1 / 1.008,
    minimumIonizationGeVPerMeter: 0.02856,
  });
  const ELECTRON_MASS = 0.00051099895;
  const K = 0.000307075; // GeV cm² mol⁻¹
  const CURVATURE = 0.299792458; // p[GeV/c] = CURVATURE B[T] R[m]

  function normal(random = Math.random) {
    return Math.sqrt(-2 * Math.log(Math.max(1e-12, random()))) * Math.cos(2 * Math.PI * random());
  }

  function densityCorrection(betaGamma) {
    const x = Math.log10(Math.max(1e-12, betaGamma));
    if (x < 0.44) return 0;
    const asymptotic = 2 * Math.LN10 * x - 3.0977;
    return x < 1.8856 ? asymptotic + 0.1348 * Math.pow(1.8856 - x, 5.6249) : asymptotic;
  }

  // Collision losses only. A low-velocity continuation avoids extending the
  // Bethe singularity below its validity range; endpoint ranges are approximate.
  function stoppingPower(particle, momentum) {
    if (!particle.charge || momentum <= 0) return 0;
    const mass = particle.mass;
    const energy = Math.hypot(mass, momentum);
    const beta2 = Math.pow(momentum / energy, 2);
    const gamma = energy / mass;
    const electron = mass < 0.001;
    const betaFloor = electron ? 0.05 : 0.02;
    if (beta2 < betaFloor * betaFloor) {
      const referenceMomentum = mass * betaFloor / Math.sqrt(1 - betaFloor * betaFloor);
      return stoppingPower(particle, referenceMomentum) * Math.sqrt(beta2) / betaFloor;
    }
    const delta = densityCorrection(momentum / mass);
    let bracket;
    let factor = K * HYDROGEN.zOverA * particle.charge * particle.charge / beta2;
    if (electron) {
      const tau = gamma - 1;
      bracket = Math.log(ELECTRON_MASS * ELECTRON_MASS * beta2 * gamma * gamma * tau /
        (2 * HYDROGEN.excitationEnergyGeV * HYDROGEN.excitationEnergyGeV));
      if (particle.charge < 0) {
        bracket += 1 - beta2 - (2 * gamma - 1) * Math.LN2 / (gamma * gamma) +
          Math.pow(tau / gamma, 2) / 8;
      } else {
        const u = gamma + 1;
        bracket += 2 * Math.LN2 - beta2 * (23 + 14 / u + 10 / (u * u) + 4 / (u * u * u)) / 12;
      }
      bracket -= delta;
      factor *= 0.5;
    } else {
      const ratio = ELECTRON_MASS / mass;
      const maximumTransfer = 2 * ELECTRON_MASS * beta2 * gamma * gamma /
        (1 + 2 * gamma * ratio + ratio * ratio);
      bracket = 0.5 * Math.log(2 * ELECTRON_MASS * beta2 * gamma * gamma * maximumTransfer /
        (HYDROGEN.excitationEnergyGeV * HYDROGEN.excitationEnergyGeV)) - beta2 - delta / 2;
    }
    return Math.max(0, factor * bracket) * HYDROGEN.densityGramsPerCubicCentimeter * 100;
  }

  // Angular diffusion: variance is additive in physical path length. Applying
  // Highland's logarithmic correction independently at every tiny step would
  // make the result depend on the integration resolution.
  function scatteringVariance(particle, momentum, distanceMeters) {
    if (!particle.charge || momentum <= 0) return 0;
    const beta = momentum / Math.hypot(particle.mass, momentum);
    return Math.pow(0.0136 * particle.charge / (beta * momentum), 2) *
      distanceMeters / HYDROGEN.radiationLengthMeters;
  }

  function transport(particle, start, initialMomentum, options = {}) {
    const {
      pixelsPerMeter = 360, stepPixels = 3.5, magneticFieldTesla = 1.4,
      visualMillisecondsPerMeterAtC = 800, maximumTransportSteps = 14000,
      maximumPathMeters = Infinity, random = Math.random, scattering = true,
      stopping = true, isInside = () => true,
    } = options;
    let x = start.x;
    let y = start.y;
    let momentum = Math.hypot(initialMomentum.px, initialMomentum.py);
    let angle = Math.atan2(initialMomentum.py, initialMomentum.px);
    let elapsed = 0;
    let properLength = 0;
    let pathLength = 0;
    let entered = isInside(start);
    let stopReason = "transport-limit";
    const points = [{ x, y }];
    const times = [0];
    const states = [];
    const snapshot = () => ({ x, y, px: momentum * Math.cos(angle), py: momentum * Math.sin(angle),
      properLength, pathLength, stoppingPower: stoppingPower(particle, momentum) });
    states.push(snapshot());
    for (let index = 1; index <= maximumTransportSteps; index += 1) {
      const energy = Math.hypot(particle.mass, momentum);
      const kinetic = energy - particle.mass;
      if (kinetic < 0.00002 || momentum < 0.00002) { stopReason = "ranged-out"; break; }
      const beta = momentum / energy;
      const loss = stopping ? stoppingPower(particle, momentum) : 0;
      const curvature = CURVATURE * particle.charge * magneticFieldTesla / momentum;
      const distance = Math.min(stepPixels / pixelsPerMeter,
        Math.abs(curvature) > 0 ? 0.045 / Math.abs(curvature) : Infinity,
        loss > 0 ? 0.04 * kinetic / loss : Infinity, maximumPathMeters - pathLength);
      if (distance < 1e-9) { stopReason = "path-limit"; break; }
      // Exact constant-curvature arc, with scattering split across the step.
      const kick = scattering ? normal(random) * Math.sqrt(scatteringVariance(particle, momentum, distance)) : 0;
      const startAngle = angle + kick / 2;
      const bend = curvature * distance;
      const halfBend = bend / 2;
      const arcScale = Math.abs(halfBend) < 1e-8 ? 1 : Math.sin(halfBend) / halfBend;
      x += distance * pixelsPerMeter * arcScale * Math.cos(startAngle + halfBend);
      y += distance * pixelsPerMeter * arcScale * Math.sin(startAngle + halfBend);
      angle += bend + kick;
      pathLength += distance;
      properLength += distance * particle.mass / momentum;
      elapsed += visualMillisecondsPerMeterAtC * distance / Math.max(beta, 0.03);
      const nextEnergy = Math.max(particle.mass, energy - loss * distance);
      momentum = Math.sqrt(Math.max(0, (nextEnergy - particle.mass) * (nextEnergy + particle.mass)));
      points.push({ x, y }); times.push(elapsed); states.push(snapshot());
      const inside = isInside({ x, y });
      entered ||= inside;
      // The viewport is a cropped view. Keep a modest physical margin before
      // retiring exiting tracks; no artificial re-entry after leaving the liquid.
      if (entered && !isInside({ x, y }, 80)) { stopReason = "exited"; break; }
    }
    return { particle, points, times, states, stopReason };
  }

  // Integrate a Poisson optical-depth threshold along path length, carrying it
  // across segment boundaries. The drawing resolution does not set bubble count.
  function sampleBubbles(track, { pixelsPerMeter = 360, random = Math.random, ratePerMeter = 260 } = {}) {
    const bubbles = [];
    let remaining = -Math.log(Math.max(1e-12, 1 - random()));
    for (let index = 1; index < track.points.length; index += 1) {
      const first = track.points[index - 1];
      const last = track.points[index];
      const state = track.states?.[index - 1];
      const distance = track.states ? track.states[index].pathLength - state.pathLength :
        Math.hypot(last.x - first.x, last.y - first.y) / pixelsPerMeter;
      const ionization = state?.stoppingPower || HYDROGEN.minimumIonizationGeVPerMeter;
      const relative = Math.min(4, Math.max(0.6, ionization / HYDROGEN.minimumIonizationGeVPerMeter));
      const rate = ratePerMeter * relative;
      let offset = 0;
      while (remaining <= (distance - offset) * rate) {
        offset += remaining / rate;
        const fraction = distance > 0 ? offset / distance : 0;
        bubbles.push({
          x: first.x + (last.x - first.x) * fraction + normal(random) * 0.09,
          y: first.y + (last.y - first.y) * fraction + normal(random) * 0.09,
          time: track.delay + track.times[index - 1] + (track.times[index] - track.times[index - 1]) * fraction,
          radius: (0.28 + 0.23 * random()) * Math.pow(relative, 0.18),
          strength: 0.55 + 0.45 * random(),
        });
        remaining = -Math.log(Math.max(1e-12, 1 - random()));
      }
      remaining -= (distance - offset) * rate;
    }
    return bubbles;
  }

  // Start a collimated ray just upstream of a rectangular chamber view. Angles
  // use canvas coordinates; offset is transverse to the beam through its centre.
  function beamEntry(width, height, angle, offset = 0, margin = 18) {
    const dx = Math.cos(angle);
    const dy = Math.sin(angle);
    const x = width / 2 - dy * offset;
    const y = height / 2 + dx * offset;
    const toX = Math.abs(dx) < 1e-10 ? Infinity : (dx > 0 ? x : width - x) / Math.abs(dx);
    const toY = Math.abs(dy) < 1e-10 ? Infinity : (dy > 0 ? y : height - y) / Math.abs(dy);
    const upstream = Math.min(toX, toY) + margin;
    return { x: x - dx * upstream, y: y - dy * upstream, angle };
  }

  return Object.freeze({ HYDROGEN, normal, densityCorrection, stoppingPower,
    scatteringVariance, transport, sampleBubbles, beamEntry });
});
