import { buildMergeRank, encodeText } from "../js/bpe/encoder.js";
import { mountPlayground } from "../js/bpe/playground.js";

const LANG_COLORS = { en: "#5b9fd4", hi: "#e85d75", te: "#3ecf8e", ta: "#f0b429" };
const ORDER = ["en", "hi", "te", "ta"];

let data = null;

function download(filename, content, mime = "text/plain") {
  const blob = new Blob([content], { type: mime });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

function renderScore(stats) {
  document.getElementById("score-value").textContent = stats.score;
  document.getElementById("score-formula").innerHTML =
    `1000 / (${stats.x_max} − ${stats.x_min}) = <strong>${stats.score}</strong>`;

  const maxR = stats.x_max;
  const bars = document.getElementById("ratio-bars");
  bars.innerHTML = ORDER.map((lang) => {
    const r = stats.ratios[lang];
    const pct = Math.min(100, (r / maxR) * 100);
    const label = stats.x_labels[lang];
    return `<div class="ratio-row">
      <span>${label}</span>
      <div class="ratio-track"><div class="ratio-fill" style="width:${pct}%;background:${LANG_COLORS[lang]}"></div></div>
      <span>${r.toFixed(4)}</span>
    </div>`;
  }).join("");
}

function renderTable(stats) {
  const tbody = document.querySelector("#stats-table tbody");
  tbody.innerHTML = ORDER.map((lang) => {
    const d = stats.details[lang];
    return `<tr>
      <td><strong>${d.language}</strong> (${d.ratio_symbol})</td>
      <td>${d.wiki_title}</td>
      <td>${d.word_count.toLocaleString()}</td>
      <td>${d.char_count.toLocaleString()}</td>
      <td>${d.token_count.toLocaleString()}</td>
      <td>${d.vocab_allocated.toLocaleString()}</td>
      <td><strong>${d.ratio.toFixed(4)}</strong></td>
    </tr>`;
  }).join("");
}

function renderSorted(stats) {
  document.getElementById("sorted-list").innerHTML = stats.sorted_desc
    .map(
      (row) =>
        `<li><strong>${row.symbol}</strong> ${row.name} — ${row.ratio.toFixed(4)}</li>`
    )
    .join("");
}

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function renderVocab(vocab, filter = "") {
  const grid = document.getElementById("vocab-grid");
  const q = filter.toLowerCase();
  const shown = q ? vocab.filter((t) => t.toLowerCase().includes(q)) : vocab;
  document.getElementById("vocab-count").textContent = `${shown.length} / ${vocab.length} tokens`;
  grid.innerHTML = shown
    .slice(0, 2000)
    .map((t, i) => `<div class="vocab-cell" title="${escapeHtml(t)}">${i}: ${escapeHtml(t)}</div>`)
    .join("");
  if (shown.length > 2000) {
    grid.innerHTML += `<div class="vocab-cell muted">… ${shown.length - 2000} more (download full list)</div>`;
  }
}

async function verify(mergeRank, stats) {
  const out = document.getElementById("verify-out");
  out.textContent = "Verifying…\n";
  const lines = [];
  let ok = true;
  const liveRatios = {};

  for (const lang of ORDER) {
    const res = await fetch(`corpora/${lang}.txt`);
    const text = await res.text();
    const tokens = encodeText(text, mergeRank);
    const vocab = stats.details[lang].vocab_allocated;
    const claimed = stats.details[lang].token_count;
    const ratio = tokens.length / vocab;
    liveRatios[lang] = ratio;
    const match = tokens.length === claimed;
    if (!match) ok = false;
    lines.push(
      `${lang.toUpperCase()}: tokens=${tokens.length} (claimed ${claimed}) ${match ? "✓" : "✗ MISMATCH"}`,
      `  ratio = ${tokens.length}/${vocab} = ${ratio.toFixed(4)} (claimed ${stats.ratios[lang].toFixed(4)})`,
      `  words = ${(text.match(/\S+/g) || []).length}`
    );
  }

  const vals = ORDER.map((l) => liveRatios[l]);
  const xMax = Math.max(...vals);
  const xMin = Math.min(...vals);
  const liveScore = (1000 / (xMax - xMin)).toFixed(2);

  lines.push(
    "",
    `X_max = ${xMax.toFixed(4)}, X_min = ${xMin.toFixed(4)}, spread = ${(xMax - xMin).toFixed(4)}`,
    `Live score = 1000 / spread = ${liveScore} (claimed ${stats.score})`,
    ok ? "\nAll token counts match claimed values." : "\nWARNING: mismatch detected."
  );
  out.textContent = lines.join("\n");
}

async function init() {
  const res = await fetch("tokenizer.json");
  data = await res.json();
  const { stats, vocab, merges } = data;
  const mergeRank = buildMergeRank(merges);
  const vocabSize = data.meta?.total_vocab ?? vocab.length;

  renderScore(stats);
  renderTable(stats);
  renderSorted(stats);
  renderVocab(vocab);

  mountPlayground(document.getElementById("bpe-playground"), {
    mergeRank,
    vocab,
    vocabSize,
  });

  document.getElementById("vocab-search").addEventListener("input", (e) => {
    renderVocab(vocab, e.target.value);
  });

  document.getElementById("verify-btn").addEventListener("click", () => verify(mergeRank, stats));

  document.getElementById("download-json").addEventListener("click", () => {
    download("tokenizer.json", JSON.stringify(data, null, 2), "application/json");
  });

  document.getElementById("download-vocab").addEventListener("click", () => {
    download("vocab.txt", vocab.map((t, i) => `${i}\t${t}`).join("\n"));
  });

  document.getElementById("download-merges").addEventListener("click", () => {
    const csv = ["rank,language,left,right,merged", ...merges.map((m) =>
      [m.rank, m.language, JSON.stringify(m.left), JSON.stringify(m.right), JSON.stringify(m.merged)].join(",")
    )].join("\n");
    download("merges.csv", csv);
  });
}

init().catch((err) => {
  document.body.innerHTML = `<pre style="color:#f66;padding:2rem">Failed to load: ${err}</pre>`;
});
