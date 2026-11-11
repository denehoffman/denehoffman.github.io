(() => {
  "use strict";

  const demo = document.querySelector("[data-laddu-spectrum]");
  if (!demo) return;

  const toggle = demo.querySelector("[data-laddu-interference-toggle]");
  const state = demo.querySelector("[data-laddu-interference-state]");
  const totalLabel = demo.querySelector("[data-laddu-total-label]");
  const description = demo.querySelector("[data-laddu-spectrum-description]");
  const caption = demo.querySelector("[data-laddu-spectrum-caption]");
  const inputs = {
    magnitudeOne: demo.querySelector('[data-laddu-input="magnitude-one"]'),
    magnitudeTwo: demo.querySelector('[data-laddu-input="magnitude-two"]'),
    phase: demo.querySelector('[data-laddu-input="phase"]'),
  };
  const outputs = {
    magnitudeOne: demo.querySelector('[data-laddu-output="magnitude-one"]'),
    magnitudeTwo: demo.querySelector('[data-laddu-output="magnitude-two"]'),
    phase: demo.querySelector('[data-laddu-output="phase"]'),
  };
  const equations = {
    coherent: demo.querySelector('[data-laddu-equation="coherent"]'),
    incoherent: demo.querySelector('[data-laddu-equation="incoherent"]'),
  };
  const paths = {
    one: demo.querySelector('[data-laddu-spectrum="one"]'),
    two: demo.querySelector('[data-laddu-spectrum="two"]'),
    total: demo.querySelector('[data-laddu-spectrum="total"]'),
  };

  const binCount = 48;
  const massMinimum = 1.0;
  const massMaximum = 1.8;
  const plot = { left: 60, right: 655, top: 25, bottom: 265 };
  let interferenceEnabled = true;

  function amplitude(mass, poleMass, width, strength = 1) {
    const real = poleMass ** 2 - mass ** 2;
    const imaginary = -poleMass * width;
    const denominator = real ** 2 + imaginary ** 2;
    return {
      real: (strength * real) / denominator,
      imaginary: (-strength * imaginary) / denominator,
    };
  }

  function intensity(value) {
    return value.real ** 2 + value.imaginary ** 2;
  }

  function rotate(value, angle) {
    return {
      real: value.real * Math.cos(angle) - value.imaginary * Math.sin(angle),
      imaginary: value.real * Math.sin(angle) + value.imaginary * Math.cos(angle),
    };
  }

  function add(left, right) {
    return {
      real: left.real + right.real,
      imaginary: left.imaginary + right.imaginary,
    };
  }

  function xForBoundary(index) {
    return plot.left + (index / binCount) * (plot.right - plot.left);
  }

  function yForValue(value) {
    const verticalMaximum = 5;
    return plot.bottom - (Math.min(value, verticalMaximum) / verticalMaximum) * (plot.bottom - plot.top) * 0.92;
  }

  function histogramPath(values, closeArea = false) {
    let path = `M ${xForBoundary(0)} ${closeArea ? plot.bottom : yForValue(values[0])}`;
    if (closeArea) path += ` L ${xForBoundary(0)} ${yForValue(values[0])}`;
    values.forEach((value, index) => {
      path += ` H ${xForBoundary(index + 1)} V ${yForValue(values[index + 1] ?? value)}`;
    });
    if (closeArea) path += ` L ${xForBoundary(binCount)} ${plot.bottom} Z`;
    return path;
  }

  function update() {
    const magnitudeOne = Number(inputs.magnitudeOne.value);
    const magnitudeTwo = Number(inputs.magnitudeTwo.value);
    const phaseDegrees = Number(inputs.phase.value);
    const phase = (phaseDegrees * Math.PI) / 180;
    const bins = Array.from({ length: binCount }, (_, index) => {
      const mass = massMinimum + ((index + 0.5) / binCount) * (massMaximum - massMinimum);
      const amplitudeOne = amplitude(mass, 1.3, 0.12, 0.16 * magnitudeOne);
      const amplitudeTwo = rotate(amplitude(mass, 1.48, 0.2, 0.2 * magnitudeTwo), phase);
      const one = intensity(amplitudeOne);
      const two = intensity(amplitudeTwo);
      return {
        one,
        two,
        coherent: intensity(add(amplitudeOne, amplitudeTwo)),
        incoherent: one + two,
      };
    });
    const totalKey = interferenceEnabled ? "coherent" : "incoherent";
    paths.one.setAttribute("d", histogramPath(bins.map((bin) => bin.one), true));
    paths.two.setAttribute("d", histogramPath(bins.map((bin) => bin.two), true));
    paths.total.setAttribute("d", histogramPath(bins.map((bin) => bin[totalKey])));
    toggle.setAttribute("aria-checked", String(interferenceEnabled));
    state.textContent = interferenceEnabled ? "On" : "Off";
    totalLabel.textContent = interferenceEnabled ? "Coherent total" : "Incoherent total";
    equations.coherent.hidden = !interferenceEnabled;
    equations.incoherent.hidden = interferenceEnabled;
    outputs.magnitudeOne.value = `${magnitudeOne.toFixed(2)}×`;
    outputs.magnitudeTwo.value = `${magnitudeTwo.toFixed(2)}×`;
    outputs.phase.value = `${phaseDegrees}°`;
    description.textContent = `The individual resonance distributions and their ${interferenceEnabled ? "coherent sum with interference enabled" : "incoherent sum with interference disabled"}.`;
    caption.textContent = interferenceEnabled
      ? `With interference on, the total includes the cross term at a relative phase of ${phaseDegrees}°.`
      : "With interference off, the total is the direct sum of the two component intensities.";
  }

  toggle.addEventListener("click", () => {
    interferenceEnabled = !interferenceEnabled;
    update();
  });
  Object.values(inputs).forEach((input) => input.addEventListener("input", update));

  update();
})();
