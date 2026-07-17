import {
  checkRoundtrip,
  encodeSpans,
  tokenLabel,
  DEFAULT_PLAYGROUND_TEXT,
  ROUNDTRIP_SAMPLE,
} from "./hf-tokenizer.js";

function escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderTokenSpans(spans) {
  return spans
    .map((span, i) => {
      const cls = `token-c${i % 12}`;
      const label = tokenLabel(span.text);
      const tip = `id: ${span.id} · &quot;${escapeHtml(label)}&quot;`;
      const visible = escapeHtml(span.text);
      return `<span class="token-highlight ${cls}" title="${tip}" aria-label="token id ${span.id}">
        ${visible}<span class="token-highlight-tip">id: ${span.id} · &quot;${escapeHtml(label)}&quot;</span>
      </span>`;
    })
    .join("");
}

function computeStats(text, tokenizer, vocabSize) {
  const spans = encodeSpans(tokenizer, text);
  const encodeTokens = spans.length;
  const uniqueTokens = new Set(spans.map((s) => s.text)).size;
  const chars = text.length;
  const roundtrip = checkRoundtrip(tokenizer, text);
  const sampleRt = checkRoundtrip(tokenizer, ROUNDTRIP_SAMPLE);

  return {
    encodeTokens,
    uniqueTokens,
    chars,
    vocabSize,
    roundtripOk: roundtrip.ok,
    sampleRoundtripOk: sampleRt.ok,
  };
}

function statsLine(s) {
  const rt = s.roundtripOk ? "roundtrip ✓" : "roundtrip ✗";
  const sample = s.sampleRoundtripOk ? "sample ✓" : "sample ✗";
  return `${s.encodeTokens} tokens · ${s.uniqueTokens} unique · ${s.chars} chars · ${s.vocabSize.toLocaleString()} vocab · ${rt} · ${sample}`;
}

export function mountPlayground(root, { tokenizer, vocabSize = 10000, initialText }) {
  if (!root) return;

  const sample = initialText ?? DEFAULT_PLAYGROUND_TEXT;

  root.innerHTML = `
    <div class="nn-demo bpe-playground" data-section="tokenizer-playground">
      <p class="bpe-playground-sub">Shared <strong>10k</strong> BPE vocab with Metaspace decode — punctuation, URLs, and number separators round-trip faithfully.</p>
      <div class="bpe-playground-stats" data-out="playground-stats"></div>
      <div class="nn-panels bpe-playground-panels">
        <div class="nn-panel bpe-playground-panel">
          <div class="nn-panel-title">Input</div>
          <textarea class="bpe-playground-input" data-input="playground-text" rows="8" spellcheck="false">${escapeHtml(sample)}</textarea>
        </div>
        <div class="nn-panel bpe-playground-panel">
          <div class="nn-panel-title">Tokenized</div>
          <div class="bpe-token-view" data-out="playground-output"></div>
        </div>
      </div>
    </div>`;

  const textarea = root.querySelector("[data-input=playground-text]");
  const statsEl = root.querySelector("[data-out=playground-stats]");
  const outputEl = root.querySelector("[data-out=playground-output]");

  function update() {
    const text = textarea.value;
    const stats = computeStats(text, tokenizer, vocabSize);
    statsEl.textContent = statsLine(stats);
    outputEl.innerHTML = renderTokenSpans(encodeSpans(tokenizer, text));
  }

  textarea.addEventListener("input", update);
  update();
}

export async function mountPlaygroundFromUrl(root, tokenizerUrl) {
  const { loadTokenizer } = await import("./hf-tokenizer.js");
  let tokenizer;
  try {
    tokenizer = await loadTokenizer(tokenizerUrl);
  } catch (err) {
    console.error(err);
    if (root) {
      root.innerHTML = `<p class="muted">Playground unavailable — could not load <code>tokenizer.json</code> from <code>${tokenizerUrl}</code>.</p>`;
    }
    return;
  }
  mountPlayground(root, {
    tokenizer,
    vocabSize: tokenizer.getVocabSize(),
  });
}
