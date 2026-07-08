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
    widget: `${root}bpe-assignment/`,
    tokenizer: `${root}bpe-assignment/tokenizer.json`,
  };
}

export function wireWidgetLink(linkEl) {
  if (!linkEl) return;
  const { widget } = getBpePaths();
  linkEl.href = widget;
  linkEl.textContent = "open full widget →";
}
