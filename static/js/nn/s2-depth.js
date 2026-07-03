import {
  loadD3,
  getThemeColors,
  generateRings,
  makeLinearModel,
  makeDeepLinear,
  makeDeepRelu,
  predictLinear,
  predictDeepLinear,
  predictDeepRelu,
  stepLinear,
  stepDeepLinear,
  stepDeepRelu,
  drawDecisionPanel,
  watchTheme,
} from "./shared.js";

export function initS2Depth(root) {
  if (root.dataset.initialized) return;
  root.dataset.initialized = "1";

  root.innerHTML = `
    <div class="nn-controls">
      <button type="button" class="nn-btn nn-train">Train all three</button>
      <button type="button" class="nn-btn nn-reset">Reset</button>
    </div>
    <div class="nn-panels nn-panels-3">
      <div class="nn-panel"><div class="nn-panel-plot" data-id="a"></div><div class="nn-acc" data-id="a-acc">—</div></div>
      <div class="nn-panel"><div class="nn-panel-plot" data-id="b"></div><div class="nn-acc" data-id="b-acc">—</div></div>
      <div class="nn-panel"><div class="nn-panel-plot" data-id="c"></div><div class="nn-acc" data-id="c-acc">—</div></div>
    </div>
    <p class="nn-hint">Models A and B should match — five linear layers collapse to one straight boundary. ReLU breaks the tie.</p>
  `;

  let d3;
  let pts = generateRings(0.08);
  let modelA = makeLinearModel();
  let modelB = makeDeepLinear(5, 8);
  let modelC = makeDeepRelu(5, 8);
  let epoch = 0;
  let animId = null;

  const el = (sel) => root.querySelector(sel);

  function reset() {
    pts = generateRings(0.08);
    modelA = makeLinearModel();
    modelB = makeDeepLinear(5, 8);
    modelC = makeDeepRelu(5, 8);
    epoch = 0;
    render();
  }

  function render() {
    const colors = getThemeColors();
    const specs = [
      { id: "a", title: "1 linear layer", pred: (x, y) => predictLinear(modelA, x, y), accId: "a-acc" },
      { id: "b", title: "5 linear layers", pred: (x, y) => predictDeepLinear(modelB, x, y), accId: "b-acc" },
      { id: "c", title: "5 layers + ReLU", pred: (x, y) => predictDeepRelu(modelC, x, y), accId: "c-acc" },
    ];
    for (const s of specs) {
      const r = drawDecisionPanel(d3, d3.select(el(`[data-id="${s.id}"]`)), s.pred, pts, s.title, colors);
      el(`[data-id="${s.accId}"]`).textContent = `Accuracy: ${(r.acc * 100).toFixed(0)}%`;
    }
  }

  function trainStep() {
    const lr = 0.25;
    stepLinear(modelA, pts, lr);
    stepDeepLinear(modelB, pts, lr);
    stepDeepRelu(modelC, pts, lr);
    epoch++;
    render();
    if (epoch < 500) animId = requestAnimationFrame(trainStep);
    else animId = null;
  }

  loadD3().then((d3m) => {
    d3 = d3m;
    render();
    watchTheme(() => render());
    el(".nn-train").addEventListener("click", () => {
      if (animId) cancelAnimationFrame(animId);
      epoch = 0;
      animId = requestAnimationFrame(trainStep);
    });
    el(".nn-reset").addEventListener("click", () => {
      if (animId) cancelAnimationFrame(animId);
      reset();
    });
  });
}
