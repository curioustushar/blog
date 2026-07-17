import { checkRoundtrip, encodeSpans, ROUNDTRIP_SAMPLE } from "../js/bpe/hf-tokenizer.js";
import { mountPlayground } from "../js/bpe/playground.js";

const ORDER = ["en", "hi", "te", "ta"];

let tokenizer = null;
let report = null;

function download(filename, content, mime = "text/plain") {
  const blob = new Blob([content], { type: mime });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

function renderTable(stats) {
  const tbody = document.querySelector("#stats-table tbody");
  tbody.innerHTML = ORDER.map((lang) => {
    const d = stats.details[lang];
    return `<tr>
      <td><strong>${d.language}</strong> (${d.ratio_symbol})</td>
      <td>${d.wiki_title}</td>
      <td>${d.faithful_units.toLocaleString()}</td>
      <td>${d.char_count.toLocaleString()}</td>
      <td>${d.token_count.toLocaleString()}</td>
      <td>10,000 shared</td>
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

function renderVocab(tokenizer, filter = "") {
  const grid = document.getElementById("vocab-grid");
  const vocab = tokenizer.getVocab(true);
  const entries = Object.entries(vocab).sort((a, b) => a[1] - b[1]);
  const q = filter.toLowerCase();
  const shown = q ? entries.filter(([tok]) => tok.toLowerCase().includes(q)) : entries;
  document.getElementById("vocab-count").textContent = `${shown.length} / ${entries.length} tokens`;
  grid.innerHTML = shown
    .slice(0, 2000)
    .map(([tok, id]) => `<div class="vocab-cell" title="${escapeHtml(tok)}">${id}: ${escapeHtml(tok)}</div>`)
    .join("");
  if (shown.length > 2000) {
    grid.innerHTML += `<div class="vocab-cell muted">… ${shown.length - 2000} more (download full list)</div>`;
  }
}

async function verify(tokenizer, stats) {
  const out = document.getElementById("verify-out");
  out.textContent = "Verifying…\n";
  const lines = [];
  let ok = true;

  const sampleRt = checkRoundtrip(tokenizer, ROUNDTRIP_SAMPLE);
  lines.push(
    `Roundtrip gate: decode(encode(${JSON.stringify(ROUNDTRIP_SAMPLE)}))`,
    `  visible chars preserved: ${sampleRt.ok ? "✓" : "✗ FAILED"}`,
    `  decoded: ${JSON.stringify(sampleRt.decoded)}`,
    ""
  );
  if (!sampleRt.ok) ok = false;

  for (const lang of ORDER) {
    const res = await fetch(`corpus/${lang}.faithful.txt`);
    const text = await res.text();
    const tokens = encodeSpans(tokenizer, text);
    const claimed = stats.details[lang].token_count;
    const match = tokens.length === claimed;
    if (!match) ok = false;
    const rt = checkRoundtrip(tokenizer, text.slice(0, 5000));
    lines.push(
      `${lang.toUpperCase()}: tokens=${tokens.length} (claimed ${claimed}) ${match ? "✓" : "✗ MISMATCH"}`,
      `  fertility = ${tokens.length}/${stats.details[lang].faithful_units} = ${(tokens.length / stats.details[lang].faithful_units).toFixed(4)}`,
      `  roundtrip (first 5k chars): ${rt.ok ? "✓" : "✗"}`
    );
  }

  const vals = ORDER.map((l) => stats.ratios[l]);
  const xMax = Math.max(...vals);
  const xMin = Math.min(...vals);
  const liveScore = (1000 / (xMax - xMin)).toFixed(2);

  lines.push(
    "",
    `X_max = ${xMax.toFixed(4)}, X_min = ${xMin.toFixed(4)}, spread = ${(xMax - xMin).toFixed(4)}`,
    `Live score = 1000 / spread = ${liveScore} (claimed ${stats.score})`,
    ok ? "\nAll checks passed." : "\nWARNING: mismatch detected."
  );
  out.textContent = lines.join("\n");
}

async function init() {
  const [tokRes, reportRes] = await Promise.all([
    fetch("tokenizer.json"),
    fetch("report.json"),
  ]);
  const tokJson = await tokRes.text();
  report = await reportRes.json();
  const { tokenizerFromJSON } = await import("../js/bpe/hf-tokenizer.js");
  tokenizer = tokenizerFromJSON(tokJson);

  const stats = report;
  const vocabSize = tokenizer.getVocabSize();

  renderTable(stats);
  renderSorted(stats);
  renderVocab(tokenizer);

  mountPlayground(document.getElementById("bpe-playground"), {
    tokenizer,
    vocabSize,
  });

  document.getElementById("vocab-search").addEventListener("input", (e) => {
    renderVocab(tokenizer, e.target.value);
  });

  document.getElementById("verify-btn").addEventListener("click", () => verify(tokenizer, stats));

  const shareUrl = new URL("tokenizer.json", window.location.href).href;
  const shareInput = document.getElementById("tokenizer-share-url");
  const directLink = document.getElementById("tokenizer-json-direct");
  if (shareInput) shareInput.value = shareUrl;
  if (directLink) directLink.href = shareUrl;

  document.getElementById("copy-share-url")?.addEventListener("click", async () => {
    const btn = document.getElementById("copy-share-url");
    try {
      await navigator.clipboard.writeText(shareUrl);
      if (btn) {
        const prev = btn.textContent;
        btn.textContent = "Copied!";
        setTimeout(() => { btn.textContent = prev; }, 2000);
      }
    } catch {
      shareInput?.select();
      document.execCommand("copy");
    }
  });

  document.getElementById("download-json").addEventListener("click", () => {
    download("tokenizer.json", tokJson, "application/json");
  });

  document.getElementById("download-vocab").addEventListener("click", () => {
    const vocab = tokenizer.getVocab(true);
    const lines = Object.entries(vocab)
      .sort((a, b) => a[1] - b[1])
      .map(([tok, id]) => `${id}\t${tok}`);
    download("vocab.txt", lines.join("\n"));
  });

  document.getElementById("download-merges")?.addEventListener("click", () => {
    download("report.json", JSON.stringify(report, null, 2), "application/json");
  });
}

init().catch((err) => {
  document.body.innerHTML = `<pre style="color:#f66;padding:2rem">Failed to load: ${err}</pre>`;
});
