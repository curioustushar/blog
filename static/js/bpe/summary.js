import { mountPlaygroundFromUrl } from "./playground.js";
import { getBpePaths, wireWidgetLink } from "./paths.js";

const { widget: WIDGET_URL, tokenizer: TOKENIZER_URL } = getBpePaths();

export async function mountBpeSummary(el) {
  if (!el) return;
  try {
    const res = await fetch(TOKENIZER_URL);
    const { stats } = await res.json();
    const rows = stats.sorted_desc
      .map(
        (r) =>
          `<tr><td>${r.symbol}</td><td>${r.name}</td><td><strong>${r.ratio.toFixed(4)}</strong></td></tr>`
      )
      .join("");

    el.innerHTML = `
      <div class="bpe-score-card">
        <div class="bpe-score-value">${stats.score}</div>
        <p class="bpe-score-label">Self score · 1000 / (${stats.x_max} − ${stats.x_min})</p>
        <table class="bpe-mini-table">
          <thead><tr><th>Ratio</th><th>Language</th><th>Value</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>`;
  } catch {
    el.innerHTML = `<p class="muted">Score summary unavailable.</p>`;
  }
}

const summaryEl = document.getElementById("bpe-summary");
if (summaryEl) mountBpeSummary(summaryEl);

const playgroundEl = document.getElementById("bpe-playground");
if (playgroundEl) mountPlaygroundFromUrl(playgroundEl, TOKENIZER_URL);

wireWidgetLink(document.getElementById("bpe-full-widget-link"));
