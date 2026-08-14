+++
title = "laddu"
[extra]
scripts = ["laddu-interference.js"]
+++
<!-- markdownlint-disable MD034 -->
{{ project_header(project="laddu") }}

[`laddu`](https://github.com/denehoffman/laddu) (/ˈlʌduː/) is a library for analysis of particle physics data. It is intended to be a simple and efficient alternative to some of the other tools out there. `laddu` is written in Rust with bindings to Python via `PyO3` and `maturin` and is the spiritual successor to `rustitude`, one of my first Rust projects. The goal of this project is to allow users to perform complex amplitude analyses (like partial-wave analyses) without complex code or configuration files.

<section class="laddu-spectrum" data-laddu-spectrum>
  <header class="laddu-spectrum__header">
    <div>
      <p class="laddu-spectrum__kicker">A small amplitude-analysis idea</p>
      <h2>When resonances overlap</h2>
    </div>
    <p class="laddu-spectrum__note">Illustrative Breit–Wigner amplitudes—not a live <code>laddu</code> calculation.</p>
  </header>
  <p class="laddu-spectrum__intro">Two resonances can contribute to the same mass spectrum. Their amplitudes add before the intensity is calculated, so the observed distribution is not necessarily just the two shapes stacked together.</p>
  <div class="laddu-spectrum__toolbar">
    <ul class="laddu-spectrum__legend" aria-label="Histogram series">
      <li><i class="laddu-spectrum__key laddu-spectrum__key--one"></i><span>Resonance 1 <small>1.30 GeV · Γ = 0.12 GeV</small></span></li>
      <li><i class="laddu-spectrum__key laddu-spectrum__key--two"></i><span>Resonance 2 <small>1.48 GeV · Γ = 0.20 GeV</small></span></li>
      <li><i class="laddu-spectrum__key laddu-spectrum__key--total"></i><span data-laddu-total-label>Coherent total</span></li>
    </ul>
    <button class="laddu-spectrum__toggle" type="button" role="switch" aria-checked="true" data-laddu-interference-toggle>
      <span>Interference</span>
      <strong data-laddu-interference-state>On</strong>
    </button>
  </div>
  <div class="laddu-spectrum__controls">
    <label>
      <span>Resonance 1 magnitude <output data-laddu-output="magnitude-one">1.00×</output></span>
      <input data-laddu-input="magnitude-one" type="range" min="0" max="1.5" step="0.01" value="1">
    </label>
    <label>
      <span>Resonance 2 magnitude <output data-laddu-output="magnitude-two">1.00×</output></span>
      <input data-laddu-input="magnitude-two" type="range" min="0" max="1.5" step="0.01" value="1">
    </label>
    <label>
      <span>Relative phase <output data-laddu-output="phase">55°</output></span>
      <input data-laddu-input="phase" type="range" min="-180" max="180" step="1" value="55">
    </label>
  </div>
  <div class="laddu-spectrum__equation" aria-live="polite">
    <span>Calculated intensity</span>
    <p data-laddu-equation="coherent"><var>I</var>(m) = | c₁ BW₁(m) + e<sup>iφ</sup> c₂ BW₂(m) |<sup>2</sup></p>
    <p data-laddu-equation="incoherent" hidden><var>I</var>(m) = | c₁ BW₁(m) |<sup>2</sup> + | c₂ BW₂(m) |<sup>2</sup></p>
  </div>
  <figure class="laddu-spectrum__plot">
    <svg viewBox="0 0 680 330" role="img" aria-labelledby="laddu-spectrum-title laddu-spectrum-description">
      <title id="laddu-spectrum-title">Mass distribution of two overlapping resonances</title>
      <desc id="laddu-spectrum-description" data-laddu-spectrum-description>The individual resonance distributions and their coherent sum with interference enabled.</desc>
      <g class="laddu-spectrum__grid" aria-hidden="true">
        <line x1="60" y1="265" x2="655" y2="265"></line>
        <line x1="60" y1="195" x2="655" y2="195"></line>
        <line x1="60" y1="125" x2="655" y2="125"></line>
        <line x1="60" y1="55" x2="655" y2="55"></line>
      </g>
      <g class="laddu-spectrum__axes" aria-hidden="true">
        <line x1="60" y1="25" x2="60" y2="265"></line>
        <line x1="60" y1="265" x2="655" y2="265"></line>
        <text x="58" y="286">1.0</text>
        <text x="207" y="286">1.2</text>
        <text x="356" y="286">1.4</text>
        <text x="505" y="286">1.6</text>
        <text x="651" y="286">1.8</text>
        <text class="laddu-spectrum__x-label" x="357" y="316">Mass (GeV)</text>
        <text class="laddu-spectrum__y-label" transform="translate(18 170) rotate(-90)">Intensity (arbitrary units)</text>
      </g>
      <path class="laddu-spectrum__area laddu-spectrum__area--one" data-laddu-spectrum="one"></path>
      <path class="laddu-spectrum__area laddu-spectrum__area--two" data-laddu-spectrum="two"></path>
      <path class="laddu-spectrum__total" data-laddu-spectrum="total"></path>
    </svg>
  </figure>
  <p class="laddu-spectrum__caption" data-laddu-spectrum-caption>With interference on, the total includes the cross term between the two amplitudes.</p>
</section>

`laddu` grew out of my frustration with the way amplitude analyses were being done within the GlueX collaboration. Everyone had these messy configuration files which would need to be duplicated and modified, usually by one-off scripts, to produce fit results which would then have to be collected by yet another set of scripts. I got tired of the constant file management, I was spending more time debugging config files than actually doing physics! Since my original foray into Rust in March 2024, I have learned a lot about what is required to distribute a project like this via Python. There are tons of small optimizations that can be made, and I'd imagine there are still quite a few to go. This project has taught me everything from memory management to the intricacies of floating-point numbers to quite a lot about parallel processing. I believe the project is still in an exploratory state, but it is certainly usable enough to do some actual research now. Since I'm actively using it while I develop it, I quickly discover new sharp corners and quality-of-life features to implement, and there's always little chores to do like documentation and testing.
