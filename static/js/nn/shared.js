/** Shared helpers for neural-network interactive demos */

export const D3_URL = "https://cdn.jsdelivr.net/npm/d3@7/+esm";

let d3Promise = null;
export function loadD3() {
  if (!d3Promise) d3Promise = import(D3_URL);
  return d3Promise;
}

export function sigmoid(z) {
  if (z >= 0) return 1 / (1 + Math.exp(-z));
  const e = Math.exp(z);
  return e / (1 + e);
}

export const relu = (z) => (z > 0 ? z : 0);

export function randn() {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function getThemeColors() {
  const s = getComputedStyle(document.documentElement);
  return {
    class0: s.getPropertyValue("--class-a").trim() || "#2d6a6a",
    class1: s.getPropertyValue("--class-b").trim() || "#c45c3e",
    text: s.getPropertyValue("--text").trim() || "#1a1a1a",
    muted: s.getPropertyValue("--text-muted").trim() || "#5c5c5c",
    border: s.getPropertyValue("--border").trim() || "#e2dfd8",
    bg: s.getPropertyValue("--bg-elevated").trim() || "#fff",
    accent: s.getPropertyValue("--accent").trim() || "#2d6a6a",
  };
}

export function generateRings(noise = 0.08, n = 300) {
  const pts = [];
  const nPer = Math.floor(n / 2);
  for (let c = 0; c < 2; c++) {
    const radius = c === 0 ? 0.5 : 1.0;
    for (let i = 0; i < nPer; i++) {
      const a = Math.random() * Math.PI * 2;
      pts.push({
        x: radius * Math.cos(a) + randn() * noise,
        y: radius * Math.sin(a) + randn() * noise,
        label: c,
      });
    }
  }
  return pts;
}

export function accuracy(predict, pts) {
  let ok = 0;
  for (const p of pts) {
    const pred = predict(p.x, p.y) >= 0.5 ? 1 : 0;
    if (pred === p.label) ok++;
  }
  return ok / pts.length;
}

export function makeLinearModel() {
  return { w0: randn() * 0.2, w1: randn() * 0.2, b: 0 };
}

export function predictLinear(m, x, y) {
  return sigmoid(m.w0 * x + m.w1 * y + m.b);
}

export function stepLinear(m, pts, lr) {
  let gw0 = 0;
  let gw1 = 0;
  let gb = 0;
  let loss = 0;
  for (const p of pts) {
    const yhat = predictLinear(m, p.x, p.y);
    const err = yhat - p.label;
    gw0 += err * p.x;
    gw1 += err * p.y;
    gb += err;
    const y = p.label;
    loss -= y * Math.log(yhat + 1e-8) + (1 - y) * Math.log(1 - yhat + 1e-8);
  }
  const n = pts.length;
  m.w0 -= (lr * gw0) / n;
  m.w1 -= (lr * gw1) / n;
  m.b -= (lr * gb) / n;
  return loss / n;
}

export function makeReluModel(hidden = 16) {
  return {
    hidden,
    w1: Array.from({ length: hidden }, () => [randn() * 0.4, randn() * 0.4]),
    b1: new Array(hidden).fill(0),
    w2: Array.from({ length: hidden }, () => randn() * 0.4),
    b2: 0,
  };
}

export function predictRelu(m, x, y) {
  let z = m.b2;
  for (let j = 0; j < m.hidden; j++) {
    z += m.w2[j] * relu(m.w1[j][0] * x + m.w1[j][1] * y + m.b1[j]);
  }
  return sigmoid(z);
}

export function stepRelu(m, pts, lr) {
  let loss = 0;
  for (const p of pts) {
    const h = new Array(m.hidden);
    const a = new Array(m.hidden);
    for (let j = 0; j < m.hidden; j++) {
      h[j] = m.w1[j][0] * p.x + m.w1[j][1] * p.y + m.b1[j];
      a[j] = relu(h[j]);
    }
    let z = m.b2;
    for (let j = 0; j < m.hidden; j++) z += m.w2[j] * a[j];
    const yhat = sigmoid(z);
    const err = yhat - p.label;
    loss -= p.label * Math.log(yhat + 1e-8) + (1 - p.label) * Math.log(1 - yhat + 1e-8);

    let dz = err * yhat * (1 - yhat);
    for (let j = 0; j < m.hidden; j++) {
      const da = dz * m.w2[j] * (h[j] > 0 ? 1 : 0);
      m.w2[j] -= lr * dz * a[j];
      m.w1[j][0] -= lr * da * p.x;
      m.w1[j][1] -= lr * da * p.y;
      m.b1[j] -= lr * da;
    }
    m.b2 -= lr * dz;
  }
  return loss / pts.length;
}

/** Stack of linear layers (collapses to one effective map) */
export function makeDeepLinear(depth = 5, width = 8) {
  const layers = [];
  let inDim = 2;
  for (let i = 0; i < depth - 1; i++) {
    layers.push({
      W: Array.from({ length: width }, () =>
        Array.from({ length: inDim }, () => randn() * 0.2)
      ),
      b: new Array(width).fill(0),
      out: width,
      in: inDim,
    });
    inDim = width;
  }
  layers.push({
    W: [Array.from({ length: inDim }, () => randn() * 0.2)],
    b: [0],
    out: 1,
    in: inDim,
  });
  return { layers, depth, width };
}

function matVec(W, b, x) {
  return W.map((row, i) => row.reduce((s, w, j) => s + w * x[j], 0) + b[i]);
}

export function predictDeepLinear(m, x, y) {
  let h = [x, y];
  for (const layer of m.layers) {
    h = matVec(layer.W, layer.b, h);
  }
  return sigmoid(h[0]);
}

export function stepDeepLinear(m, pts, lr) {
  let loss = 0;
  for (const p of pts) {
    const acts = [[p.x, p.y]];
    for (const layer of m.layers) {
      acts.push(matVec(layer.W, layer.b, acts[acts.length - 1]));
    }
    const yhat = sigmoid(acts[acts.length - 1][0]);
    const err = yhat - p.label;
    loss -= p.label * Math.log(yhat + 1e-8) + (1 - p.label) * Math.log(1 - yhat + 1e-8);

    let delta = [err * yhat * (1 - yhat)];
    for (let L = m.layers.length - 1; L >= 0; L--) {
      const layer = m.layers[L];
      const prev = acts[L];
      const dPrev = new Array(prev.length).fill(0);
      for (let i = 0; i < layer.out; i++) {
        for (let j = 0; j < layer.in; j++) {
          layer.W[i][j] -= lr * delta[i] * prev[j];
          dPrev[j] += delta[i] * layer.W[i][j];
        }
        layer.b[i] -= lr * delta[i];
      }
      delta = dPrev;
    }
  }
  return loss / pts.length;
}

export function makeDeepRelu(depth = 5, width = 8) {
  const layers = [];
  let inDim = 2;
  for (let i = 0; i < depth - 1; i++) {
    layers.push({
      W: Array.from({ length: width }, () =>
        Array.from({ length: inDim }, () => randn() * 0.3)
      ),
      b: new Array(width).fill(0),
      out: width,
      in: inDim,
    });
    inDim = width;
  }
  layers.push({
    W: [Array.from({ length: inDim }, () => randn() * 0.3)],
    b: [0],
    out: 1,
    in: inDim,
  });
  return { layers, depth, width };
}

export function predictDeepRelu(m, x, y) {
  let h = [x, y];
  for (let i = 0; i < m.layers.length - 1; i++) {
    const layer = m.layers[i];
    h = matVec(layer.W, layer.b, h).map(relu);
  }
  const last = m.layers[m.layers.length - 1];
  h = matVec(last.W, last.b, h);
  return sigmoid(h[0]);
}

export function stepDeepRelu(m, pts, lr) {
  let loss = 0;
  for (const p of pts) {
    const cache = [];
    let h = [p.x, p.y];
    cache.push({ h: [...h], layer: null });
    for (let i = 0; i < m.layers.length; i++) {
      const layer = m.layers[i];
      const z = matVec(layer.W, layer.b, h);
      const isLast = i === m.layers.length - 1;
      h = isLast ? z : z.map(relu);
      cache.push({ z, h: [...h], layer, isLast, pre: cache[cache.length - 1].h });
    }
    const yhat = sigmoid(h[0]);
    const err = yhat - p.label;
    loss -= p.label * Math.log(yhat + 1e-8) + (1 - p.label) * Math.log(1 - yhat + 1e-8);

    let delta = [err * yhat * (1 - yhat)];
    for (let i = m.layers.length - 1; i >= 0; i--) {
      const { layer, pre, z, isLast } = cache[i + 1];
      const dZ = isLast ? delta : delta.map((d, j) => d * (z[j] > 0 ? 1 : 0));
      const dPrev = new Array(pre.length).fill(0);
      for (let o = 0; o < layer.out; o++) {
        for (let j = 0; j < layer.in; j++) {
          layer.W[o][j] -= lr * dZ[o] * pre[j];
          dPrev[j] += dZ[o] * layer.W[o][j];
        }
        layer.b[o] -= lr * dZ[o];
      }
      delta = dPrev;
    }
  }
  return loss / pts.length;
}

export function drawDecisionPanel(d3, container, predict, pts, title, colors) {
  const W = 280;
  const H = 260;
  const pad = 28;
  container.selectAll("*").remove();

  const svg = container
    .append("svg")
    .attr("viewBox", `0 0 ${W} ${H}`)
    .attr("class", "nn-panel-svg");

  const xExt = d3.extent(pts, (d) => d.x);
  const yExt = d3.extent(pts, (d) => d.y);
  const mx = (xExt[1] - xExt[0]) * 0.35 || 0.3;
  const my = (yExt[1] - yExt[0]) * 0.35 || 0.3;
  const xScale = d3.scaleLinear().domain([xExt[0] - mx, xExt[1] + mx]).range([pad, W - pad]);
  const yScale = d3.scaleLinear().domain([yExt[0] - my, yExt[1] + my]).range([H - pad, pad]);

  const gridN = 48;
  const xs = d3.range(gridN).map((i) => xScale.domain()[0] + (i / (gridN - 1)) * (xScale.domain()[1] - xScale.domain()[0]));
  const ys = d3.range(gridN).map((i) => yScale.domain()[0] + (i / (gridN - 1)) * (yScale.domain()[1] - yScale.domain()[0]));
  const values = new Array(gridN * gridN);
  for (let j = 0; j < gridN; j++) {
    for (let i = 0; i < gridN; i++) {
      values[j * gridN + i] = predict(xs[i], ys[j]);
    }
  }

  const contours = d3.contours().size([gridN, gridN]).thresholds(12);
  const geoPath = d3.geoPath();

  const c0 = d3.color(colors.class0);
  const c1 = d3.color(colors.class1);

  svg
    .selectAll("path.field")
    .data(contours(values))
    .join("path")
    .attr("class", "field")
    .attr("d", geoPath)
    .attr("transform", `translate(${pad},${pad}) scale(${(W - 2 * pad) / gridN},${(H - 2 * pad) / gridN})`)
    .attr("fill", (d) => {
      const t = d.value;
      const col = d3.interpolateRgb(c0.copy({ opacity: 0.12 }), c1.copy({ opacity: 0.35 }))(t);
      return col;
    })
    .attr("stroke", "none");

  svg
    .selectAll("path.boundary")
    .data(d3.contours().size([gridN, gridN]).thresholds([0.5])(values))
    .join("path")
    .attr("class", "boundary")
    .attr("d", geoPath)
    .attr("transform", `translate(${pad},${pad}) scale(${(W - 2 * pad) / gridN},${(H - 2 * pad) / gridN})`)
    .attr("fill", "none")
    .attr("stroke", colors.text)
    .attr("stroke-width", 2)
    .attr("opacity", 0.85);

  svg
    .selectAll("circle.pt")
    .data(pts)
    .join("circle")
    .attr("class", "pt")
    .attr("cx", (d) => xScale(d.x))
    .attr("cy", (d) => yScale(d.y))
    .attr("r", 3.5)
    .attr("fill", (d) => (d.label === 0 ? colors.class0 : colors.class1))
    .attr("stroke", colors.bg)
    .attr("stroke-width", 0.8);

  svg
    .append("text")
    .attr("x", pad)
    .attr("y", 16)
    .attr("fill", colors.text)
    .attr("font-size", 12)
    .attr("font-weight", 600)
    .text(title);

  return { acc: accuracy(predict, pts) };
}

export function watchTheme(callback) {
  const obs = new MutationObserver(callback);
  obs.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  return obs;
}
