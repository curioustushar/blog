/**
 * Resolve blog-relative URLs from the summary script tag (works local + production).
 */
export function getBpePaths() {
  const script = document.currentScript;
  if (script?.dataset?.widgetUrl) {
    const widget = script.dataset.widgetUrl;
    const tokenizer =
      script.dataset.tokenizerUrl ||
      `${widget.replace(/\/?$/, "/")}tokenizer.json`;
    return { widget, tokenizer };
  }

  const base = script?.src?.replace(/\/js\/bpe\/summary\.js.*$/, "") || "/blog";
  const root = base.replace(/\/?$/, "/");
  return {
    widget: `${root}bpe/`,
    tokenizer: `${root}bpe/tokenizer.json`,
  };
}

export function wireWidgetLink(linkEl) {
  if (!linkEl) return;
  const { widget } = getBpePaths();
  linkEl.href = widget;
  linkEl.textContent = "open full widget →";
}

export function wireTokenizerDownloadLink(linkEl, { showUrl = false } = {}) {
  if (!linkEl) return;
  const { tokenizer } = getBpePaths();
  linkEl.href = tokenizer;
  linkEl.setAttribute("download", "tokenizer.json");
  if (showUrl) linkEl.textContent = tokenizer;
}
