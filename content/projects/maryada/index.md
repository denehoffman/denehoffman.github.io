+++
title = "maryada"
[extra]
scripts = ["maryada-branch-and-bound.js"]
+++
<!-- markdownlint-disable MD034 -->
{{ project_header(project="maryada") }}

[`maryada`](https://github.com/denehoffman/maryada) is a `no_std` binary64 interval arithmetic library conforming to IEEE Std 1788.1-2017. It provides bare and decorated real intervals, outward-rounded elementary operations, and text and binary interchange. Rectangular complex intervals are available through an optional feature.

The conformance claim applies to the real interval API. The rectangular complex interval extension is outside the scope of the standard. The repository's [conformance statement](https://github.com/denehoffman/maryada/blob/main/CONFORMANCE.md) documents the operation accuracy declarations, required features, implementation details, and test coverage.

## Finding the deeper well

Interval arithmetic is especially useful when a numerical optimizer needs a global guarantee rather than a promising local answer. The explorer below applies branch-and-bound to a one-dimensional function with two local minima. Each branch is an interval `X`; evaluating the formula with interval operations produces an enclosure that contains every possible function value on that branch.

<section class="maryada-bnb" data-maryada-bnb aria-labelledby="maryada-bnb-title">
  <header class="maryada-bnb__header">
    <div>
      <p class="maryada-bnb__kicker">Interval arithmetic in practice</p>
      <h2 id="maryada-bnb-title">Can we certify the deeper well?</h2>
    </div>
    <p class="maryada-bnb__note">Browser illustration · stop when <var>U − L</var> ≤ 0.03</p>
  </header>

  <p class="maryada-bnb__intro">The left minimum is lower, but a local search started on the right would not know that. Branch-and-bound keeps the best sampled value <var>U</var> and compares it with interval lower bounds <var>L</var>. Click a branch to inspect its bounds, then step or play through the search.</p>

  <div class="maryada-bnb__formula" aria-label="Problem setup">
    <div>
      <span>Objective</span>
      <p><var>f</var>(x) = (x<sup>2</sup> − 1)<sup>2</sup> + 0.6x + 2</p>
    </div>
    <div>
      <span>Search domain</span>
      <p><var>X</var> = [−1.5, 1.5]</p>
    </div>
    <div>
      <span>Discard test</span>
      <p><var>L</var> ≥ <var>U</var> − 0.03</p>
    </div>
  </div>

  <div class="maryada-bnb__controls" role="group" aria-label="Branch-and-bound controls">
    <div class="maryada-bnb__actions">
      <button type="button" data-maryada-reset>Reset</button>
      <button class="colored" type="button" data-maryada-step>Take next step <span aria-hidden="true">→</span></button>
      <button type="button" data-maryada-play aria-pressed="false">Play</button>
    </div>
    <p class="maryada-bnb__status" data-maryada-status role="status" aria-live="polite">Step 0 · The root interval is ready to split.</p>
  </div>

  <dl class="maryada-bnb__stats" aria-label="Search status">
    <div>
      <dt>Best sample <var>U</var></dt>
      <dd data-maryada-incumbent>3.000</dd>
    </div>
    <div>
      <dt>Lowest leaf <var>L</var></dt>
      <dd data-maryada-lower>1.100</dd>
    </div>
    <div>
      <dt>Gap <var>U − L</var></dt>
      <dd data-maryada-gap>1.900</dd>
    </div>
    <div>
      <dt>Active / pruned</dt>
      <dd data-maryada-frontier>1 / 0</dd>
    </div>
  </dl>

  <div class="maryada-bnb__workbench">
    <figure class="maryada-bnb__plot">
      <svg viewBox="0 0 760 500" role="group" aria-labelledby="maryada-bnb-plot-title maryada-bnb-plot-description">
        <title id="maryada-bnb-plot-title">Branch-and-bound search over a two-well function</title>
        <desc id="maryada-bnb-plot-description" data-maryada-description>Interval enclosures for the root branch and the current best sampled point.</desc>
        <defs>
          <clipPath id="maryada-bnb-plot-clip">
            <rect x="56" y="25" width="674" height="390"></rect>
          </clipPath>
        </defs>
        <g class="maryada-bnb__grid" aria-hidden="true">
          <line x1="56" y1="399" x2="730" y2="399"></line>
          <line x1="56" y1="313" x2="730" y2="313"></line>
          <line x1="56" y1="228" x2="730" y2="228"></line>
          <line x1="56" y1="142" x2="730" y2="142"></line>
          <line x1="56" y1="57" x2="730" y2="57"></line>
        </g>
        <g class="maryada-bnb__bounds" data-maryada-bounds clip-path="url(#maryada-bnb-plot-clip)"></g>
        <path class="maryada-bnb__curve" data-maryada-curve clip-path="url(#maryada-bnb-plot-clip)"></path>
        <line class="maryada-bnb__incumbent-line" data-maryada-incumbent-line clip-path="url(#maryada-bnb-plot-clip)" x1="56" x2="730"></line>
        <circle class="maryada-bnb__incumbent-dot" data-maryada-incumbent-dot r="5"></circle>
        <text class="maryada-bnb__incumbent-label" data-maryada-incumbent-label></text>
        <g class="maryada-bnb__axes" aria-hidden="true">
          <line x1="56" y1="25" x2="56" y2="415"></line>
          <line x1="56" y1="415" x2="730" y2="415"></line>
          <text x="44" y="403">1.2</text>
          <text x="44" y="317">2.0</text>
          <text x="44" y="232">2.8</text>
          <text x="44" y="146">3.6</text>
          <text x="44" y="61">4.4</text>
          <text x="56" y="438">−1.5</text>
          <text x="168" y="438">−1</text>
          <text x="280" y="438">−0.5</text>
          <text x="393" y="438">0</text>
          <text x="505" y="438">0.5</text>
          <text x="617" y="438">1</text>
          <text x="730" y="438">1.5</text>
          <text class="maryada-bnb__x-label" x="393" y="480">x</text>
          <text class="maryada-bnb__y-label" transform="translate(15 220) rotate(-90)">f(x)</text>
        </g>
      </svg>
      <figcaption>Each translucent box is an interval enclosure <var>f(X)</var>; the curve is clipped above 4.7 so both wells remain legible.</figcaption>
      <ul class="maryada-bnb__legend" aria-label="Plot legend">
        <li><i class="maryada-bnb__key maryada-bnb__key--active"></i>unresolved branch</li>
        <li><i class="maryada-bnb__key maryada-bnb__key--pruned"></i>discarded branch</li>
        <li><i class="maryada-bnb__key maryada-bnb__key--candidate"></i>best sample <var>U</var></li>
      </ul>
    </figure>
    <aside class="maryada-bnb__inspector" aria-labelledby="maryada-bnb-inspector-title">
      <p class="maryada-bnb__inspector-kicker">Branch inspector</p>
      <h3 id="maryada-bnb-inspector-title">Inspect <span data-maryada-selected-label>B0</span></h3>
      <p class="maryada-bnb__selected-path" data-maryada-selected-path>root interval</p>
      <dl>
        <div>
          <dt>Interval evaluation</dt>
          <dd><code data-maryada-selected-enclosure>f(X) ⊆ [1.100, 4.463]</code></dd>
        </div>
        <div>
          <dt>Lower bound <var>L</var></dt>
          <dd data-maryada-selected-lower>1.100</dd>
        </div>
        <div>
          <dt>Midpoint sample</dt>
          <dd data-maryada-selected-sample>x = 0.000 → f(x) = 3.000</dd>
        </div>
        <div>
          <dt>Disposition</dt>
          <dd data-maryada-selected-status>Active: the branch can still contain a better point.</dd>
        </div>
      </dl>
      <p class="maryada-bnb__inspector-note" data-maryada-selected-note>Splitting this interval replaces it with two smaller enclosures.</p>
    </aside>
  </div>

  <div class="maryada-bnb__ledger">
    <div class="maryada-bnb__ledger-header">
      <div>
        <p class="maryada-bnb__inspector-kicker">Search history</p>
        <h3>Branch ledger</h3>
      </div>
      <p>Click any branch or box to inspect its numbers.</p>
    </div>
    <div class="maryada-bnb__table-wrap">
      <table>
        <thead>
          <tr>
            <th scope="col">Branch</th>
            <th scope="col"><var>X</var></th>
            <th scope="col"><var>f(X)</var> enclosure</th>
            <th scope="col">midpoint sample</th>
            <th scope="col">state</th>
          </tr>
        </thead>
        <tbody data-maryada-branches>
          <tr>
            <th scope="row"><button type="button" data-maryada-branch-select="branch-0" aria-pressed="true">B0</button></th>
            <td><code>[−1.500, 1.500]</code></td>
            <td><code>[1.100, 4.463]</code></td>
            <td><code>x = 0.000 → 3.000</code></td>
            <td><span class="maryada-bnb__state maryada-bnb__state--active">active</span></td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>

  <p class="maryada-bnb__caption" data-maryada-caption>At each step, the branch with the smallest lower bound is split. The current candidate is the midpoint sample at x = 0.000.</p>
</section>

The important test is `L ≥ U − ε`: once a branch's lower bound is too high to improve the incumbent by more than the requested tolerance, the whole branch can be discarded. In a production implementation, outward-rounded interval operations make that enclosure rigorous; this browser illustration mirrors the same natural interval extension and rounds the displayed values for readability.

## A small interval

```rust
use maryada::Interval;

let x = Interval::new(1.0, 2.0);
let y = x.sqr();

assert_eq!(y.bounds(), (1.0, 4.0));
```

The optional `complex` feature enables rectangular complex intervals through `ComplexBox`; `num-complex` adds interoperability with `num_complex::Complex64`. Complex functions transform interval spaces in nontrivial ways, so their results contain the true image but are not guaranteed to equal it under every operation.

Future directions include other complex interval formulations such as disks and polyarcs, linear algebra methods, and potentially Python bindings.
