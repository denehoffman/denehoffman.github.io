/* Planar pi- + hydrogen production and selected visible decays.
 * GeV, c=1. Exact on-shell kinematics; illustrative channel weights/angles.
 * Lorentz transforms and phase-space factorization: PDG Kinematics, sec. 49.
 * https://pdg.lbl.gov/2025/reviews/rpp2025-rev-kinematics.pdf
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.BubbleInteractions = factory();
})(typeof globalThis === "object" ? globalThis : this, function () {
  "use strict";
  const particles = Object.freeze(Object.fromEntries(Object.entries({
    "pi-": [0.13957039, -1, 0, 0], "pi+": [0.13957039, 1, 0, 0],
    pi0: [0.1349768, 0, 0, 0], p: [0.938272088, 1, 1, 0],
    K0: [0.497611, 0, 0, 1], "K+": [0.493677, 1, 0, 1],
    Lambda: [1.115683, 0, 1, -1], "Xi-": [1.32171, -1, 1, -2],
    "e-": [0.000510999, -1, 0, 0], "e+": [0.000510999, 1, 0, 0], gamma: [0, 0, 0, 0],
  }).map(([id, [mass, charge, baryon, strangeness]]) =>
    [id, Object.freeze({ id, mass, charge, baryon, strangeness })])));
  const channels = Object.freeze({
    track: Object.freeze(["pi-", "p"]),
    decay: Object.freeze(["K0", "Lambda"]),
    cascade: Object.freeze(["K+", "K0", "Xi-"]),
    dalitz: Object.freeze(["pi-", "p", "pi0"]),
  });
  function vector(id, momentum) {
    return { id, energy: Math.hypot(particles[id].mass, momentum.px, momentum.py), px: momentum.px, py: momentum.py };
  }
  function boost(rest, total, mass) {
    const betaX = total.px / total.energy, betaY = total.py / total.energy;
    const beta2 = betaX * betaX + betaY * betaY;
    if (beta2 < 1e-24) return { ...rest };
    const gamma = total.energy / mass;
    const dot = betaX * rest.px + betaY * rest.py;
    const factor = (gamma - 1) * dot / beta2 + gamma * rest.energy;
    return { energy: gamma * (rest.energy + dot), px: rest.px + factor * betaX, py: rest.py + factor * betaY };
  }
  function restMomentum(mass, first, second) {
    if (mass < first + second) return null;
    return Math.sqrt(Math.max(0, (mass - first - second) * (mass + first + second)
      * (mass - first + second) * (mass + first - second))) / (2 * mass);
  }
  function twoBody(mass, total, masses, random) {
    const p = restMomentum(mass, masses[0], masses[1]);
    if (p === null) return null;
    const angle = random() * 2 * Math.PI;
    const px = p * Math.cos(angle), py = p * Math.sin(angle);
    return [boost({ energy: Math.hypot(masses[0], p), px, py }, total, mass),
      boost({ energy: Math.hypot(masses[1], p), px: -px, py: -py }, total, mass)];
  }
  function threeBody(mass, total, masses, random) {
    const minimum = masses[0] + masses[1], maximum = mass - masses[2];
    if (maximum < minimum) return null;
    // A true upper bound, not a scanned underestimate of the rejection weight.
    const bound = restMomentum(mass, minimum, masses[2]) * restMomentum(maximum, masses[0], masses[1]);
    let intermediate = minimum;
    let accepted = bound === 0;
    for (let attempt = 0; !accepted && attempt < 4096; attempt += 1) {
      intermediate = minimum + random() * (maximum - minimum);
      const weight = restMomentum(mass, intermediate, masses[2]) * restMomentum(intermediate, masses[0], masses[1]);
      accepted = random() * bound <= weight;
    }
    if (!accepted) throw new Error("Three-body phase-space sampling did not converge");
    const outer = twoBody(mass, total, [intermediate, masses[2]], random);
    const inner = twoBody(intermediate, outer[0], masses.slice(0, 2), random);
    return [...inner, outer[1]];
  }
  function check(initial, final, strong = false) {
    const sum = (list, key) => list.reduce((value, item) => value + item[key], 0);
    for (const key of ["energy", "px", "py"]) {
      if (Math.abs(sum(initial, key) - sum(final, key)) > 2e-9 * Math.max(1, sum(initial, "energy")))
        throw new Error(`Interaction does not conserve ${key}`);
    }
    for (const item of final) {
      if (![item.energy, item.px, item.py].every(Number.isFinite) || item.energy < 0 ||
          Math.abs(item.energy ** 2 - item.px ** 2 - item.py ** 2 - particles[item.id].mass ** 2) > 1e-8 * Math.max(1, item.energy ** 2))
        throw new Error("Interaction product is not on shell");
    }
    for (const key of strong ? ["charge", "baryon", "strangeness"] : ["charge", "baryon"]) {
      const total = (list) => list.reduce((value, item) => value + particles[item.id][key], 0);
      if (total(initial) !== total(final)) throw new Error(`Interaction does not conserve ${key}`);
    }
  }
  function produce(kind, momentum, random = Math.random) {
    if (kind === "mixed") {
      const draw = random();
      kind = draw < 0.35 ? "track" : draw < 0.65 ? "decay" : draw < 0.8 ? "cascade" : "dalitz";
    }
    if (!channels[kind]) throw new Error(`Unknown interaction ${kind}`);
    const initial = [vector("pi-", momentum), vector("p", { px: 0, py: 0 })];
    const total = { energy: initial[0].energy + initial[1].energy, px: momentum.px, py: momentum.py };
    const mass = Math.sqrt(total.energy ** 2 - total.px ** 2 - total.py ** 2);
    const ids = channels[kind];
    const masses = ids.map((id) => particles[id].mass);
    const result = ids.length === 2 ? twoBody(mass, total, masses, random) : threeBody(mass, total, masses, random);
    if (!result) return null; // no sub-threshold channel or invented energy
    const products = result.map((item, index) => ({ id: ids[index], ...item }));
    check(initial, products, true);
    return { kind, initial, products };
  }
  function decay(id, momentum, random = Math.random) {
    const parent = vector(id, momentum);
    let ids, result;
    if (id === "pi0") {
      ids = ["e-", "e+", "gamma"];
      const minimum = 2.02 * particles["e-"].mass;
      const pairMass = minimum * Math.pow(particles.pi0.mass * 0.72 / minimum, random());
      const outer = twoBody(particles.pi0.mass, parent, [pairMass, 0], random);
      result = [...twoBody(pairMass, outer[0], [particles["e-"].mass, particles["e+"].mass], random), outer[1]];
    } else {
      ids = { K0: ["pi+", "pi-"], Lambda: ["p", "pi-"], "Xi-": ["Lambda", "pi-"] }[id];
      if (!ids) return null;
      result = twoBody(particles[id].mass, parent, ids.map((key) => particles[key].mass), random);
    }
    const products = result.map((item, index) => ({ id: ids[index], ...item }));
    check([parent], products); // weak decays need not conserve strangeness
    return { parent, products };
  }
  return Object.freeze({ particles, channels, produce, decay });
});
