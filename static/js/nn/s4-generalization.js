import { loadD3, getThemeColors, generateRings, makeDeepRelu, predictDeepRelu, stepDeepRelu, watchTheme } from "./shared.js";

const SIZES = [20, 200, 2000];

function splitData(nTrain) {
  const all = generateRings(0.08, nTrain + 400);
  return { train: all.slice(0, nTrain), test: all.slice(nTrain) };
}

export function initS4Generalization(root) {
  if (root.dataset.initialized) return;
  root.dataset.initialized = "1";

  root.innerHTML = `
    <div class="nn-controls">
      <button type="button" class="nn-btn nn-train">Train all three</button>
      <button type="button" class="nn-btn nn-reset">Reset</button>
    </div>
    <div class="nn-panels nn-panels-3" id="s4-panels"></div>
    <p class="nn-hint">Same architecture, different training set sizes — watch the generalization gap shrink as data grows.</p>
  `;

  const panelsEl = () => root.querySelector("#s4-panels");
  panelsEl().innerHTML = SIZES.map(
    (n) => `
    <div class="nn-panel" data-n="${n}">
      <div class="nn-panel-title">n = ${n}</div>
      <svg class="nn-acc-chart" data-chart="${n}" viewBox="0 0 200 100"></svg>
      <div class="nn-gap" data-gap="${n}">Gap: —</div>
    </div>`
  ).join("");

  let d3;
  const state = SIZES.map((n) => ({
    n,
    data: splitData(n),
    model: makeDeepRelu(4, 32),
    history: { train: [], test: [] },
  }));
  let epoch = 0;
  let animId = null;

  const el = (sel) => root.querySelector(sel);

  function acc(model, pts) {
    let ok = 0;
    for (const p of pts) {
      const pred = predictDeepRelu(model, p.x, p.y) >= 0.5 ? 1 : 0;
      if (pred === p.label) ok++;
    }
    return pts.length ? ok / pts.length : 0;
  }

  function render() {
    const colors = getThemeColors();
    for (const s of state) {
      const svg = d3.select(el(`[data-chart="${s.n}"]`));
      svg.selectAll("*").remove();
      const pad = { l: 28, r: 8, t: 8, b: 18 };
      const W = 200;
      const H = 100;
      const nPts = Math.max(s.history.train.length, 1);
      const x = d3.scaleLinear().domain([0, Math.max(nPts - 1, 1)]).range([pad.l, W - pad.r]);
      const y = d3.scaleLinear().domain([0, 1]).range([H - pad.b, pad.t]);
      const line = d3.line().x((_, i) => x(i)).y((d) => y(d));

      if (s.history.train.length) {
        svg.append("path").attr("d", line(s.history.train)).attr("fill", "none").attr("stroke", colors.class0).attr("stroke-width", 1.5);
        svg.append("path").attr("d", line(s.history.test)).attr("fill", "none").attr("stroke", colors.class1).attr("stroke-width", 1.5);
      }
      svg.append("text").attr("x", pad.l).attr("y", H - 4).attr("fill", colors.muted).attr("font-size", 8).text("train / test acc");
      const gap = s.history.train.length
        ? ((s.history.train.at(-1) - s.history.test.at(-1)) * 100).toFixed(0)
        : "—";
      el(`[data-gap="${s.n}"]`).textContent = `Gap: ${gap}pp`;
    }
  }

  function trainStep() {
    const lr = 0.2;
    for (const s of state) {
      stepDeepRelu(s.model, s.data.train, lr);
      s.history.train.push(acc(s.model, s.data.train));
      s.history.test.push(acc(s.model, s.data.test));
    }
    epoch++;
    render();
    if (epoch < 350) animId = requestAnimationFrame(trainStep);
    else animId = null;
  }

  function reset() {
    for (const s of state) {
      s.data = splitData(s.n);
      s.model = makeDeepRelu(4, 32);
      s.history = { train: [], test: [] };
    }
    epoch = 0;
    render();
  }

  loadD3().then((d3m) => {
    d3 = d3m;
    render();
    watchTheme(() => render());
    el(".nn-train").addEventListener("click", () => {
      if (animId) cancelAnimationFrame(animId);
      epoch = 0;
      for (const s of state) s.history = { train: [], test: [] };
      animId = requestAnimationFrame(trainStep);
    });
    el(".nn-reset").addEventListener("click", () => {
      if (animId) cancelAnimationFrame(animId);
      reset();
    });
  });
}
