import { mountPlaygroundFromUrl } from "./playground.js";
import { BPE_PATHS, wireTokenizerDownloadLink, wireWidgetLink } from "./paths.js";

const TOKENIZER_URL = BPE_PATHS.tokenizer;
const REPORT_URL = BPE_PATHS.report;

function fmt(n) {
  return n.toLocaleString();
}

export async function mountBpeSummary(el) {
  if (!el) return;
  try {
    const res = await fetch(REPORT_URL);
    const stats = await res.json();

    const langs = ["en", "hi", "te", "ta"];
    const items = langs.map((lang) => {
      const details = stats.details[lang];
      return {
        lang,
        details,
        fertility: details.ratio,
      };
    });

    items.sort((a, b) => b.fertility - a.fertility);
    items.forEach((item, index) => {
      item.symbol = `X${items.length - index}`;
    });

    const rows = items
      .map((r) => {
        const pass = r.fertility <= 1.2;
        const statusClass = pass ? "status-pass" : "status-fail";
        return `<tr>
            <td class="lang"><strong>${r.symbol}</strong> ${r.details.language}</td>
            <td class="num">${fmt(r.details.faithful_units)}</td>
            <td class="num">${fmt(r.details.token_count)}</td>
            <td class="num"><strong>${r.fertility.toFixed(3)}</strong></td>
            <td class="status ${statusClass}">${pass ? "pass" : "fail"}</td>
          </tr>`;
      })
      .join("");

    el.innerHTML = `
      <div class="bpe-fertility-wrap">
        <p class="bpe-fertility-intro">
          Faithful Markdown · fertility = tokens / faithful units ·
          <code>decode(encode(text))</code> preserves visible characters.
        </p>

        <h3 class="bpe-fertility-heading">Per-language fertility</h3>
        <table class="bpe-fertility-table">
          <thead>
            <tr>
              <th>Lang</th>
              <th class="num">Units</th>
              <th class="num">Tokens</th>
              <th class="num">Xi</th>
              <th class="status">≤1.2</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div>`;
  } catch (err) {
    console.error(err);
    el.innerHTML = `<p class="muted">Live fertility table unavailable (could not load <code>report.json</code>). See the Results table above.</p>`;
  }
}

const summaryEl = document.getElementById("bpe-summary");
if (summaryEl) mountBpeSummary(summaryEl);

const playgroundEl = document.getElementById("bpe-playground");
if (playgroundEl) mountPlaygroundFromUrl(playgroundEl, TOKENIZER_URL);

wireWidgetLink(document.getElementById("bpe-full-widget-link"));
wireTokenizerDownloadLink(document.getElementById("bpe-tokenizer-download-link"), { showUrl: true });
