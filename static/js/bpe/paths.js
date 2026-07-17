/**
 * Resolve blog-relative URLs from the summary script tag (works local + production).
 * Paths are captured once at module load — not inside async callbacks.
 */
function resolveBpePaths() {
  const script = document.currentScript;
  if (script?.dataset?.widgetUrl) {
    const widget = script.dataset.widgetUrl;
    const tokenizer =
      script.dataset.tokenizerUrl ||
      `${widget.replace(/\/?$/, "/")}tokenizer.json`;
    const report =
      script.dataset.reportUrl ||
      `${widget.replace(/\/?$/, "/")}report.json`;
    return { widget, tokenizer, report };
  }

  const base = script?.src?.replace(/\/js\/bpe\/summary\.js.*$/, "") || "/blog";
  const root = base.replace(/\/?$/, "/");
  return {
    widget: `${root}bpe/`,
    tokenizer: `${root}bpe/tokenizer.json`,
    report: `${root}bpe/report.json`,
  };
}

export const BPE_PATHS = resolveBpePaths();

export function getBpePaths() {
  return BPE_PATHS;
}

export function wireWidgetLink(linkEl) {
  if (!linkEl) return;
  linkEl.href = BPE_PATHS.widget;
  linkEl.textContent = "open full widget →";
}

export function wireTokenizerDownloadLink(linkEl, { showUrl = false } = {}) {
  if (!linkEl) return;
  linkEl.href = BPE_PATHS.tokenizer;
  linkEl.setAttribute("download", "tokenizer.json");
  if (showUrl) linkEl.textContent = BPE_PATHS.tokenizer;
}
