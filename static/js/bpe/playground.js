import {
  buildMergeRank,
  buildVocabIndex,
  encodeSpans,
  tokenLabel,
  DEFAULT_PLAYGROUND_TEXT,
} from "./encoder.js";

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

function computeStats(text, mergeRank, vocabIndex, vocabSize) {
  const pretokenWords = (text.match(/\S+/g) || []).length;
  const spans = encodeSpans(text, mergeRank, vocabIndex);
  const encodeTokens = spans.length;
  const uniqueTokens = new Set(spans.map((s) => s.text)).size;
  const chars = text.length;
  const fertility =
    pretokenWords > 0 ? (encodeTokens / pretokenWords).toFixed(2) : "—";

  return { encodeTokens, uniqueTokens, pretokenWords, chars, fertility, vocabSize };
}

function statsLine(s) {
  return `${s.encodeTokens} encode tokens · ${s.uniqueTokens} unique tokens · ${s.pretokenWords} pretoken words · ${s.chars} chars · ${s.vocabSize.toLocaleString()} merged vocab · fertility ${s.fertility}`;
}

export function mountPlayground(root, { mergeRank, vocab, vocabSize = 10000, initialText }) {
  if (!root) return;

  const vocabIndex = buildVocabIndex(vocab);
  const sample = initialText ?? DEFAULT_PLAYGROUND_TEXT;

  root.innerHTML = `
    <div class="nn-demo bpe-playground" data-section="tokenizer-playground">
      <p class="bpe-playground-sub">All languages share one <strong>10k</strong> merged vocab. Type or paste text in any script — English, Hindi, Telugu, or Tamil.</p>
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
    const stats = computeStats(text, mergeRank, vocabIndex, vocabSize);
    statsEl.textContent = statsLine(stats);
    outputEl.innerHTML = renderTokenSpans(encodeSpans(text, mergeRank, vocabIndex));
  }

  textarea.addEventListener("input", update);
  update();
}

export async function mountPlaygroundFromUrl(root, tokenizerUrl) {
  const res = await fetch(tokenizerUrl);
  const data = await res.json();
  mountPlayground(root, {
    mergeRank: buildMergeRank(data.merges),
    vocab: data.vocab,
    vocabSize: data.meta?.total_vocab ?? data.vocab?.length ?? 10000,
  });
}
