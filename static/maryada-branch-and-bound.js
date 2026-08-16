(() => {
  "use strict";

  const demo = document.querySelector("[data-maryada-bnb]");
  if (!demo) return;

  const SVG_NS = "http://www.w3.org/2000/svg";
  const DOMAIN = { lower: -1.5, upper: 1.5 };
  const EPSILON = 0.03;
  const PLOT = {
    left: 56,
    right: 730,
    top: 25,
    bottom: 415,
    yMinimum: 1.05,
    yMaximum: 4.7,
  };

  const elements = {
    actions: demo.querySelector(".maryada-bnb__actions"),
    branches: demo.querySelector("[data-maryada-branches]"),
    bounds: demo.querySelector("[data-maryada-bounds]"),
    caption: demo.querySelector("[data-maryada-caption]"),
    curve: demo.querySelector("[data-maryada-curve]"),
    description: demo.querySelector("[data-maryada-description]"),
    gap: demo.querySelector("[data-maryada-gap]"),
    incumbent: demo.querySelector("[data-maryada-incumbent]"),
    incumbentDot: demo.querySelector("[data-maryada-incumbent-dot]"),
    incumbentLabel: demo.querySelector("[data-maryada-incumbent-label]"),
    incumbentLine: demo.querySelector("[data-maryada-incumbent-line]"),
    lower: demo.querySelector("[data-maryada-lower]"),
    play: demo.querySelector("[data-maryada-play]"),
    reset: demo.querySelector("[data-maryada-reset]"),
    selectedEnclosure: demo.querySelector("[data-maryada-selected-enclosure]"),
    selectedLabel: demo.querySelector("[data-maryada-selected-label]"),
    selectedLower: demo.querySelector("[data-maryada-selected-lower]"),
    selectedNote: demo.querySelector("[data-maryada-selected-note]"),
    selectedPath: demo.querySelector("[data-maryada-selected-path]"),
    selectedSample: demo.querySelector("[data-maryada-selected-sample]"),
    selectedStatus: demo.querySelector("[data-maryada-selected-status]"),
    status: demo.querySelector("[data-maryada-status]"),
    step: demo.querySelector("[data-maryada-step]"),
    frontier: demo.querySelector("[data-maryada-frontier]"),
  };

  let nextBranchNumber = 0;
  let playbackTimer = null;
  let state = createState();

  function objective(x) {
    return (x * x - 1) ** 2 + 0.6 * x + 2;
  }

  function squareInterval(lower, upper) {
    return [
      lower <= 0 && upper >= 0 ? 0 : Math.min(lower ** 2, upper ** 2),
      Math.max(lower ** 2, upper ** 2),
    ];
  }

  function evaluateInterval(lower, upper) {
    const squaredX = squareInterval(lower, upper);
    const shifted = [squaredX[0] - 1, squaredX[1] - 1];
    const squaredShifted = squareInterval(shifted[0], shifted[1]);
    return {
      lower: squaredShifted[0] + 0.6 * lower + 2,
      upper: squaredShifted[1] + 0.6 * upper + 2,
    };
  }

  function makeBranch(lower, upper, depth, path, parentId = null) {
    const interval = evaluateInterval(lower, upper);
    const midpoint = (lower + upper) / 2;
    const branch = {
      id: `branch-${nextBranchNumber}`,
      label: `B${nextBranchNumber}`,
      lower,
      upper,
      depth,
      interval,
      midpoint,
      sample: objective(midpoint),
      path,
      parentId,
      status: "active",
      reason: "",
    };
    nextBranchNumber += 1;
    return branch;
  }

  function createState() {
    nextBranchNumber = 0;
    const root = makeBranch(DOMAIN.lower, DOMAIN.upper, 0, "root interval");
    return {
      branches: [root],
      incumbent: root.sample,
      incumbentX: root.midpoint,
      selectedId: root.id,
      steps: 0,
      complete: false,
      lastAction: "The root interval is ready to split.",
    };
  }

  function format(value) {
    return value.toFixed(3);
  }

  function getActiveBranches() {
    return state.branches.filter((branch) => branch.status === "active");
  }

  function getLeafBranches() {
    return state.branches.filter((branch) => branch.status !== "split");
  }

  function lowestLeafBound() {
    return Math.min(...getLeafBranches().map((branch) => branch.interval.lower));
  }

  function pruneEligibleBranches() {
    const threshold = state.incumbent - EPSILON;
    const pruned = [];
    getActiveBranches().forEach((branch) => {
      if (branch.interval.lower >= threshold) {
        branch.status = "pruned";
        branch.reason = `L = ${format(branch.interval.lower)} is at least U − ε = ${format(threshold)}.`;
        pruned.push(branch);
      }
    });
    return pruned;
  }

  function finishIfComplete() {
    if (getActiveBranches().length !== 0) return false;
    state.complete = true;
    stopPlayback();
    return true;
  }

  function chooseBranch() {
    return getActiveBranches().reduce((best, branch) => (
      branch.interval.lower < best.interval.lower ? branch : best
    ));
  }

  function step() {
    if (state.complete) return;

    const branch = chooseBranch();
    branch.status = "split";
    const splitPoint = branch.midpoint;
    const left = makeBranch(branch.lower, splitPoint, branch.depth + 1, `${branch.path} / left`, branch.id);
    const right = makeBranch(splitPoint, branch.upper, branch.depth + 1, `${branch.path} / right`, branch.id);
    state.branches.push(left, right);

    const previousIncumbent = state.incumbent;
    [left, right].forEach((child) => {
      if (child.sample < state.incumbent) {
        state.incumbent = child.sample;
        state.incumbentX = child.midpoint;
      }
    });

    const pruned = pruneEligibleBranches();
    state.steps += 1;
    state.selectedId = state.incumbent < previousIncumbent ? (
      left.sample <= right.sample ? left.id : right.id
    ) : branch.id;

    const improvement = state.incumbent < previousIncumbent
      ? ` The midpoint sample improves U to ${format(state.incumbent)}.`
      : " The incumbent stays unchanged.";
    const pruning = pruned.length > 0
      ? ` Discarded ${pruned.map((item) => item.label).join(", ")} because their lower bounds cannot beat U within ε.`
      : " No branch qualifies for pruning yet.";
    state.lastAction = `Step ${state.steps}: split ${branch.label} at x = ${format(splitPoint)}.${improvement}${pruning}`;
    finishIfComplete();
    render();
  }

  function startPlayback() {
    if (state.complete || playbackTimer !== null) return;
    elements.play.textContent = "Pause";
    elements.play.setAttribute("aria-pressed", "true");
    playbackTimer = window.setInterval(() => {
      step();
      if (state.complete) stopPlayback();
    }, 1200);
  }

  function stopPlayback() {
    if (playbackTimer !== null) {
      window.clearInterval(playbackTimer);
      playbackTimer = null;
    }
    elements.play.textContent = "Play";
    elements.play.setAttribute("aria-pressed", "false");
  }

  function xForValue(value) {
    return PLOT.left + ((value - DOMAIN.lower) / (DOMAIN.upper - DOMAIN.lower)) * (PLOT.right - PLOT.left);
  }

  function yForValue(value) {
    const clamped = Math.max(PLOT.yMinimum, Math.min(PLOT.yMaximum, value));
    return PLOT.bottom - ((clamped - PLOT.yMinimum) / (PLOT.yMaximum - PLOT.yMinimum)) * (PLOT.bottom - PLOT.top);
  }

  function makeSvgElement(tagName, attributes) {
    const element = document.createElementNS(SVG_NS, tagName);
    Object.entries(attributes).forEach(([name, value]) => element.setAttribute(name, value));
    return element;
  }

  function renderBounds() {
    elements.bounds.replaceChildren();
    getLeafBranches().forEach((branch) => {
      const rect = makeSvgElement("rect", {
        class: `maryada-bnb__bound maryada-bnb__bound--${branch.status}${branch.id === state.selectedId ? " is-selected" : ""}`,
        x: xForValue(branch.lower),
        y: yForValue(branch.interval.upper),
        width: Math.max(2, xForValue(branch.upper) - xForValue(branch.lower)),
        height: Math.max(3, yForValue(branch.interval.lower) - yForValue(branch.interval.upper)),
        "data-maryada-branch-select": branch.id,
        role: "button",
        tabindex: "0",
        "aria-label": `Inspect ${branch.label}, interval ${format(branch.lower)} to ${format(branch.upper)}`,
      });
      const title = makeSvgElement("title", {});
      title.textContent = `${branch.label}: X = [${format(branch.lower)}, ${format(branch.upper)}], f(X) is enclosed by [${format(branch.interval.lower)}, ${format(branch.interval.upper)}].`;
      rect.appendChild(title);
      elements.bounds.appendChild(rect);
    });
  }

  function renderCurve() {
    const samples = 180;
    let path = "";
    for (let index = 0; index <= samples; index += 1) {
      const x = DOMAIN.lower + (index / samples) * (DOMAIN.upper - DOMAIN.lower);
      path += `${index === 0 ? "M" : "L"} ${xForValue(x)} ${yForValue(objective(x))} `;
    }
    elements.curve.setAttribute("d", path.trim());
  }

  function renderIncumbent() {
    const y = yForValue(state.incumbent);
    const x = xForValue(state.incumbentX);
    elements.incumbentLine.setAttribute("y1", y);
    elements.incumbentLine.setAttribute("y2", y);
    elements.incumbentDot.setAttribute("cx", x);
    elements.incumbentDot.setAttribute("cy", y);
    elements.incumbentLabel.setAttribute("x", Math.min(PLOT.right - 4, x + 10));
    elements.incumbentLabel.setAttribute("y", Math.max(PLOT.top + 13, y - 8));
    elements.incumbentLabel.textContent = `U = ${format(state.incumbent)}`;
  }

  function renderTable() {
    elements.branches.innerHTML = state.branches.map((branch) => {
      const selected = branch.id === state.selectedId;
      const statusLabel = branch.status === "split" ? "split" : branch.status;
      const sample = `x = ${format(branch.midpoint)} → ${format(branch.sample)}`;
      return `<tr class="${selected ? "is-selected" : ""} maryada-bnb__row--${branch.status}">
        <th scope="row"><button type="button" data-maryada-branch-select="${branch.id}" aria-pressed="${selected}">${branch.label}</button></th>
        <td><code>[${format(branch.lower)}, ${format(branch.upper)}]</code></td>
        <td><code>[${format(branch.interval.lower)}, ${format(branch.interval.upper)}]</code></td>
        <td><code>${sample}</code></td>
        <td><span class="maryada-bnb__state maryada-bnb__state--${branch.status}">${statusLabel}</span></td>
      </tr>`;
    }).join("");
  }

  function renderInspector() {
    const branch = state.branches.find((item) => item.id === state.selectedId) || state.branches[0];
    const status = branch.status === "active"
      ? "Active: the branch can still contain a better point."
      : branch.status === "pruned"
        ? `Pruned: ${branch.reason}`
        : "Split: its two child intervals now cover this branch.";
    const note = branch.status === "pruned"
      ? "The lower bound is already close enough to the incumbent that this branch cannot matter at the requested tolerance."
      : branch.status === "split"
        ? "The children are tighter enclosures, so their lower bounds can guide the next choice."
        : "Splitting this interval replaces it with two smaller enclosures.";

    elements.selectedLabel.textContent = branch.label;
    elements.selectedPath.textContent = branch.path;
    elements.selectedEnclosure.textContent = `f(X) ⊆ [${format(branch.interval.lower)}, ${format(branch.interval.upper)}]`;
    elements.selectedLower.textContent = format(branch.interval.lower);
    elements.selectedSample.textContent = `x = ${format(branch.midpoint)} → f(x) = ${format(branch.sample)}`;
    elements.selectedStatus.textContent = status;
    elements.selectedNote.textContent = note;
  }

  function renderSummary() {
    const active = getActiveBranches();
    const lower = lowestLeafBound();
    const gap = Math.max(0, state.incumbent - lower);
    const prunedCount = state.branches.filter((branch) => branch.status === "pruned").length;
    elements.incumbent.textContent = format(state.incumbent);
    elements.lower.textContent = format(lower);
    elements.gap.textContent = format(gap);
    elements.frontier.textContent = `${active.length} / ${prunedCount}`;
    elements.status.textContent = state.complete
      ? `Step ${state.steps} · certified within ε = ${format(EPSILON)}: U − L = ${format(gap)}.`
      : `Step ${state.steps} · ${state.lastAction}`;
    elements.step.disabled = state.complete;
    elements.play.disabled = state.complete;
    elements.caption.textContent = state.complete
      ? `The search is complete: U = ${format(state.incumbent)} at x = ${format(state.incumbentX)}, while every remaining leaf has L ≥ ${format(lower)}. The gap is within ε = ${format(EPSILON)}.`
      : `${state.lastAction} The current best sample is at x = ${format(state.incumbentX)}.`;
    elements.description.textContent = state.complete
      ? `The branch-and-bound search is complete. The best sample is f(${format(state.incumbentX)}) = ${format(state.incumbent)}, certified within the requested tolerance.`
      : `The current best sample is f(${format(state.incumbentX)}) = ${format(state.incumbent)}. Interval boxes show the bounds for every leaf branch.`;
  }

  function render() {
    renderBounds();
    renderCurve();
    renderIncumbent();
    renderTable();
    renderInspector();
    renderSummary();
  }

  function selectBranch(branchId) {
    if (!state.branches.some((branch) => branch.id === branchId)) return;
    state.selectedId = branchId;
    render();
  }

  elements.step.addEventListener("click", () => {
    stopPlayback();
    step();
  });
  elements.reset.addEventListener("click", () => {
    stopPlayback();
    state = createState();
    render();
  });
  elements.play.addEventListener("click", () => {
    if (playbackTimer === null) startPlayback();
    else stopPlayback();
  });
  demo.addEventListener("click", (event) => {
    const target = event.target.closest("[data-maryada-branch-select]");
    if (target) selectBranch(target.dataset.maryadaBranchSelect);
  });
  demo.addEventListener("keydown", (event) => {
    if ((event.key === "Enter" || event.key === " ") && event.target.matches("[data-maryada-branch-select]")) {
      event.preventDefault();
      selectBranch(event.target.dataset.maryadaBranchSelect);
    }
  });

  render();
})();
