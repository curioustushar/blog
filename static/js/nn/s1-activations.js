import {
  loadD3,
  getThemeColors,
  generateRings,
  makeLinearModel,
  makeReluModel,
  predictLinear,
  predictRelu,
  stepLinear,
  stepRelu,
  drawDecisionPanel,
  watchTheme,
} from "./shared.js";

export function initS1Activations(root) {
  if (root.dataset.initialized) return;
  root.dataset.initialized = "1";

  root.innerHTML = `
    <div class="nn-controls">
      <button type="button" class="nn-btn nn-train">Train both</button>
      <button type="button" class="nn-btn nn-reset">Reset</button>
      <label class="nn-label">Noise
        <input type="range" class="nn-noise" min="0" max="0.2" step="0.01" value="0.08">
        <span class="nn-noise-val">0.08</span>
      </label>
      <label class="nn-label">Learning rate
        <input type="range" class="nn-lr" min="0.05" max="0.8" step="0.05" value="0.3">
        <span class="nn-lr-val">0.30</span>
      </label>
    </div>
    <div class="nn-panels">
      <div class="nn-panel"><div class="nn-panel-plot" data-id="linear"></div><div class="nn-acc" data-id="linear-acc">—</div></div>
      <div class="nn-panel"><div class="nn-panel-plot" data-id="relu"></div><div class="nn-acc" data-id="relu-acc">—</div></div>
    </div>
    <div class="nn-loss-chart"></div>
    <p class="nn-hint">Click <strong>Train both</strong> — watch the linear model stall near ~55% while ReLU wraps the ring.</p>
  `;

  let d3;
  let pts = generateRings(0.08);
  let linear = makeLinearModel();
  let relu = makeReluModel(16);
  let epoch = 0;
  let animId = null;
  let lossHistory = { linear: [], relu: [] };

  const el = (sel) => root.querySelector(sel);
  const noiseInput = el(".nn-noise");
  const lrInput = el(".nn-lr");

  function resetData() {
    const noise = +noiseInput.value;
    pts = generateRings(noise);
    linear = makeLinearModel();
    relu = makeReluModel(16);
    epoch = 0;
    lossHistory = { linear: [], relu: [] };
    render();
  }

  function render() {
    const colors = getThemeColors();
    const linPlot = d3.select(el('[data-id="linear"]'));
    const relPlot = d3.select(el('[data-id="relu"]'));
    const r1 = drawDecisionPanel(d3, linPlot, (x, y) => predictLinear(linear, x, y), pts, "Linear only", colors);
    const r2 = drawDecisionPanel(d3, relPlot, (x, y) => predictRelu(relu, x, y), pts, "One ReLU layer", colors);
    el('[data-id="linear-acc"]').textContent = `Accuracy: ${(r1.acc * 100).toFixed(0)}%`;
    el('[data-id="relu-acc"]').textContent = `Accuracy: ${(r2.acc * 100).toFixed(0)}%`;
    drawLossChart(colors);
  }

  function drawLossChart(colors) {
    const chart = d3.select(el(".nn-loss-chart"));
    chart.selectAll("*").remove();
    const W = 580;
    const H = 120;
    const pad = { t: 12, r: 12, b: 24, l: 36 };
    const svg = chart.append("svg").attr("viewBox", `0 0 ${W} ${H}`);
    const maxLen = Math.max(lossHistory.linear.length, lossHistory.relu.length, 1);
    const allLoss = [...lossHistory.linear, ...lossHistory.relu];
    const yMax = allLoss.length ? d3.max(allLoss) * 1.1 : 1;
    const x = d3.scaleLinear().domain([0, Math.max(maxLen - 1, 1)]).range([pad.l, W - pad.r]);
    const y = d3.scaleLinear().domain([0, yMax]).range([H - pad.b, pad.t]);
    const line = d3.line().x((_, i) => x(i)).y((d) => y(d));

    if (lossHistory.linear.length) {
      svg.append("path").attr("d", line(lossHistory.linear)).attr("fill", "none").attr("stroke", colors.muted).attr("stroke-width", 2).attr("stroke-dasharray", "4,3");
    }
    if (lossHistory.relu.length) {
      svg.append("path").attr("d", line(lossHistory.relu)).attr("fill", "none").attr("stroke", colors.accent).attr("stroke-width", 2);
    }
    svg.append("text").attr("x", pad.l).attr("y", 10).attr("fill", colors.muted).attr("font-size", 10).text("Loss — dashed: linear, solid: ReLU");
  }

  function trainStep() {
    const lr = +lrInput.value;
    lossHistory.linear.push(stepLinear(linear, pts, lr));
    lossHistory.relu.push(stepRelu(relu, pts, lr));
    epoch++;
    render();
    if (epoch < 400) animId = requestAnimationFrame(trainStep);
    else animId = null;
  }

  loadD3().then((d3m) => {
    d3 = d3m;
    render();
    watchTheme(() => render());

    el(".nn-train").addEventListener("click", () => {
      if (animId) cancelAnimationFrame(animId);
      epoch = 0;
      lossHistory = { linear: [], relu: [] };
      animId = requestAnimationFrame(trainStep);
    });
    el(".nn-reset").addEventListener("click", () => {
      if (animId) cancelAnimationFrame(animId);
      resetData();
    });
    noiseInput.addEventListener("input", () => {
      el(".nn-noise-val").textContent = (+noiseInput.value).toFixed(2);
      resetData();
    });
    lrInput.addEventListener("input", () => {
      el(".nn-lr-val").textContent = (+lrInput.value).toFixed(2);
    });
  });
}
