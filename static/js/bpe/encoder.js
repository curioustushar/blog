/** Shared byte-level BPE encoder (matches scripts/train_bpe_tokenizer.py). */

export function tokenDisplay(bytes) {
  const arr = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(arr);
  } catch {
    return "b:" + [...arr].map((b) => b.toString(16).padStart(2, "0")).join("_");
  }
}

export function tokenLabel(str) {
  if (str.length === 1) {
    const c = str.charCodeAt(0);
    if (c === 32) return "<0x20>";
    if (c === 10) return "<0x0A>";
    if (c === 9) return "<0x09>";
    if (c < 32 || c === 127) return `<0x${c.toString(16).toUpperCase().padStart(2, "0")}>`;
  }
  return str;
}

export function wordToSymbols(word) {
  return Array.from(new TextEncoder().encode(word)).map((b) => [b]);
}

function getPairs(symbols) {
  const pairs = new Set();
  for (let i = 0; i < symbols.length - 1; i++) {
    pairs.add(symbols[i].join(",") + "|" + symbols[i + 1].join(","));
  }
  return pairs;
}

function parseKey(key) {
  const [a, b] = key.split("|");
  return [a.split(",").map(Number), b.split(",").map(Number)];
}

export function buildMergeRank(merges) {
  const rank = new Map();
  merges.forEach((m, i) => {
    rank.set(m.left_bytes.join(",") + "|" + m.right_bytes.join(","), i);
  });
  return rank;
}

export function buildVocabIndex(vocab) {
  const map = new Map();
  vocab.forEach((tok, i) => {
    if (!map.has(tok)) map.set(tok, i);
  });
  return map;
}

export function encodeWordSymbols(word, mergeRank) {
  let symbols = wordToSymbols(word);
  let pairs = getPairs(symbols);

  while (pairs.size > 0) {
    let best = null;
    let bestRank = Infinity;
    for (const key of pairs) {
      const r = mergeRank.get(key);
      if (r !== undefined && r < bestRank) {
        best = key;
        bestRank = r;
      }
    }
    if (best === null) break;

    const [a, b] = parseKey(best);
    const merged = [...a, ...b];
    const next = [];
    let i = 0;
    while (i < symbols.length) {
      if (
        i < symbols.length - 1 &&
        symbols[i].join() === a.join() &&
        symbols[i + 1].join() === b.join()
      ) {
        next.push(merged);
        i += 2;
      } else {
        next.push(symbols[i]);
        i += 1;
      }
    }
    symbols = next;
    pairs = getPairs(symbols);
  }
  return symbols;
}

export function encodeWord(word, mergeRank) {
  return encodeWordSymbols(word, mergeRank).map((s) => tokenDisplay(s));
}

/** Corpus-level encode — whitespace skipped (matches Python trainer / claimed stats). */
export function encodeText(text, mergeRank) {
  const words = text.match(/\S+/g) || [];
  return words.flatMap((w) => encodeWord(w, mergeRank));
}

/** Playground display — preserves spaces and newlines as token spans. */
export function encodeSpans(text, mergeRank, vocabIndex) {
  const parts = text.match(/\s+|\S+/g) || [];
  const spans = [];

  for (const part of parts) {
    const syms = encodeWordSymbols(part, mergeRank);
    for (const sym of syms) {
      const display = tokenDisplay(sym);
      spans.push({
        text: display,
        id: vocabIndex.get(display) ?? -1,
      });
    }
  }
  return spans;
}

export const HIGHLIGHT_COLORS = [
  "rgba(59, 130, 246, 0.27)",
  "rgba(139, 92, 246, 0.27)",
  "rgba(236, 72, 153, 0.27)",
  "rgba(249, 115, 22, 0.27)",
  "rgba(234, 179, 8, 0.27)",
  "rgba(34, 197, 94, 0.27)",
  "rgba(20, 184, 166, 0.27)",
  "rgba(6, 182, 212, 0.27)",
  "rgba(99, 102, 241, 0.27)",
  "rgba(168, 85, 247, 0.27)",
  "rgba(244, 63, 94, 0.27)",
  "rgba(132, 204, 22, 0.27)",
];

export const DEFAULT_PLAYGROUND_TEXT = `India is multi cultural and heritage Rich

भारत बहु-सांस्कृतिक है और विरासत से समृद्ध है।

భారతదేశం బహుళ సంస్కృతిక మరియు వారసత్వ సంపన్నమైనది.

இந்தியா பல பண்பாட்டு மற்றும் பாரம்பரியம் நிறைந்தது.`;
