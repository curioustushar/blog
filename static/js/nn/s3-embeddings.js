import { loadD3, getThemeColors, watchTheme } from "./shared.js";

const TOKENS = ["cat", "dog", "cow", "apple", "mango", "eat", "chase", "see"];
const IDX = Object.fromEntries(TOKENS.map((t, i) => [t, i]));
const CAT = {
  cat: "animal", dog: "animal", cow: "animal",
  apple: "fruit", mango: "fruit",
  eat: "verb", chase: "verb", see: "verb",
};
const CAT_COLOR = { animal: "class0", fruit: "class1", verb: "verb" };

function buildPairs() {
  const pairs = [];
  const animals = ["cat", "dog", "cow"];
  const verbs = ["eat", "chase", "see"];
  const fruits = ["apple", "mango"];
  for (let i = 0; i < 6000; i++) {
    const a = animals[Math.floor(Math.random() * 3)];
    const v = verbs[Math.floor(Math.random() * 3)];
    const f = fruits[Math.floor(Math.random() * 2)];
    const sent = [a, v, f];
    for (let j = 0; j < sent.length - 1; j++) {
      pairs.push([IDX[sent[j]], IDX[sent[j + 1]]]);
    }
  }
  return pairs;
}

export function initS3Embeddings(root) {
  if (root.dataset.initialized) return;
  root.dataset.initialized = "1";

  root.innerHTML = `
    <div class="nn-controls">
      <button type="button" class="nn-btn nn-train">Train</button>
      <button type="button" class="nn-btn nn-reset">Reset</button>
      <span class="nn-step-label">Step: <span class="nn-step">0</span></span>
    </div>
    <div class="nn-embed-plot"></div>
    <p class="nn-hint">Watch clusters form — animals, fruits, and verbs group together with no similarity labels.</p>
  `;

  let d3;
  let W;
  const dim = 2;
  let pairs = buildPairs();
  let step = 0;
  let animId = null;

  const el = (sel) => root.querySelector(sel);

  function randEmbed() {
    W = TOKENS.map(() => [Math.random() * 0.2 - 0.1, Math.random() * 0.2 - 0.1]);
  }

  function trainBatch(n = 80) {
    const lr = 0.15;
    for (let k = 0; k < n; k++) {
      const [ctx, nxt] = pairs[Math.floor(Math.random() * pairs.length)];
      const logits = W.map((row) => row[0] * W[ctx][0] + row[1] * W[ctx][1]);
      const maxL = Math.max(...logits);
      const exps = logits.map((z) => Math.exp(z - maxL));
      const sum = exps.reduce((a, b) => a + b, 0);
      const probs = exps.map((e) => e / sum);
      for (let i = 0; i < TOKENS.length; i++) {
        const grad = probs[i] - (i === nxt ? 1 : 0);
        W[ctx][0] -= lr * grad * W[i][0];
        W[ctx][1] -= lr * grad * W[i][1];
        W[i][0] -= lr * grad * W[ctx][0];
        W[i][1] -= lr * grad * W[ctx][1];
      }
      step++;
    }
  }

  function render() {
    const colors = getThemeColors();
    const plot = d3.select(el(".nn-embed-plot"));
    plot.selectAll("*").remove();
    const Wd = 520;
    const Hd = 320;
    const pad = 40;
    const svg = plot.append("svg").attr("viewBox", `0 0 ${Wd} ${Hd}`);

    const xs = W.map((p) => p[0]);
    const ys = W.map((p) => p[1]);
    const xScale = d3.scaleLinear().domain(d3.extent(xs).map((v, i, a) => v + (i === 0 ? -0.15 : 0.15))).range([pad, Wd - pad]);
    const yScale = d3.scaleLinear().domain(d3.extent(ys).map((v, i, a) => v + (i === 0 ? -0.15 : 0.15))).range([Hd - pad, pad]);

    const colorMap = {
      animal: colors.class0,
      fruit: colors.class1,
      verb: getComputedStyle(document.documentElement).getPropertyValue("--verb").trim() || "#6b5b95",
    };

    svg
      .selectAll("circle")
      .data(TOKENS.map((t, i) => ({ t, i, x: W[i][0], y: W[i][1], cat: CAT[t] })))
      .join("circle")
      .attr("cx", (d) => xScale(d.x))
      .attr("cy", (d) => yScale(d.y))
      .attr("r", 10)
      .attr("fill", (d) => colorMap[d.cat])
      .attr("stroke", colors.bg)
      .attr("stroke-width", 1.5);

    svg
      .selectAll("text.lbl")
      .data(TOKENS.map((t, i) => ({ t, x: W[i][0], y: W[i][1] })))
      .join("text")
      .attr("class", "lbl")
      .attr("x", (d) => xScale(d.x) + 12)
      .attr("y", (d) => yScale(d.y) + 4)
      .attr("fill", colors.text)
      .attr("font-size", 11)
      .text((d) => d.t);

    el(".nn-step").textContent = String(step);
  }

  function animate() {
    trainBatch(12);
    render();
    if (step < 3000) animId = requestAnimationFrame(animate);
    else animId = null;
  }

  loadD3().then((d3m) => {
    d3 = d3m;
    randEmbed();
    render();
    watchTheme(() => render());
    el(".nn-train").addEventListener("click", () => {
      if (animId) cancelAnimationFrame(animId);
      animId = requestAnimationFrame(animate);
    });
    el(".nn-reset").addEventListener("click", () => {
      if (animId) cancelAnimationFrame(animId);
      step = 0;
      pairs = buildPairs();
      randEmbed();
      render();
    });
  });
}
