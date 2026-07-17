/** Pure-JS HuggingFace BPE + Metaspace runtime (no CDN). Reads tokenizer.json. */

export function visibleNonWs(text) {
  return text.replace(/\s/g, "");
}

export function tokenLabel(str) {
  if (str === "▁") return "<space>";
  if (str.length === 1) {
    const c = str.charCodeAt(0);
    if (c === 32) return "<0x20>";
    if (c === 10) return "<0x0A>";
    if (c === 9) return "<0x09>";
    if (c < 32 || c === 127) return `<0x${c.toString(16).toUpperCase().padStart(2, "0")}>`;
  }
  return str;
}

export const ROUNDTRIP_SAMPLE = "India's population is 1,428,627,663.";

export const DEFAULT_PLAYGROUND_TEXT = `India's population is 1,428,627,663.

India is multi cultural and heritage Rich

भारत बहु-सांस्कृतिक है और विरासत से समृद्ध है।

భారతదేశం బహుళ సంస్కృతిక మరియు వారసత్వ సంపన్నమైనది.

இந்தியா பல பண்பாட்டு மற்றும் பாரம்பரியம் நிறைந்தது.`;

function pairKey(left, right) {
  return `${left}\0${right}`;
}

function buildMergeRank(merges) {
  const rank = new Map();
  merges.forEach(([left, right], i) => {
    rank.set(pairKey(left, right), i);
  });
  return rank;
}

function buildVocabMaps(vocab) {
  const tokenToId = new Map(Object.entries(vocab));
  const idToToken = new Map();
  for (const [tok, id] of tokenToId.entries()) {
    idToToken.set(Number(id), tok);
  }
  return { tokenToId, idToToken };
}

function nfkc(text) {
  return text.normalize("NFKC");
}

/** Metaspace pretokenizer (prepend_scheme=never, split on ASCII space only). */
function metaspaceSegments(text) {
  const parts = text.split(" ");
  return parts.map((part, i) => (i === 0 ? part : `▁${part}`));
}

function bpeSegment(segment, mergeRank) {
  if (!segment) return [];
  let symbols = Array.from(segment);
  while (symbols.length > 1) {
    let bestRank = Infinity;
    let bestIdx = -1;
    for (let i = 0; i < symbols.length - 1; i += 1) {
      const r = mergeRank.get(pairKey(symbols[i], symbols[i + 1]));
      if (r !== undefined && r < bestRank) {
        bestRank = r;
        bestIdx = i;
      }
    }
    if (bestIdx === -1) break;
    symbols = [
      ...symbols.slice(0, bestIdx),
      symbols[bestIdx] + symbols[bestIdx + 1],
      ...symbols.slice(bestIdx + 2),
    ];
  }
  return symbols;
}

export class BrowserTokenizer {
  constructor(config) {
    this.config = config;
    this.mergeRank = buildMergeRank(config.model.merges || []);
    const { tokenToId, idToToken } = buildVocabMaps(config.model.vocab || {});
    this.tokenToId = tokenToId;
    this.idToToken = idToToken;
    this.unkId = tokenToId.get(config.model.unk_token || "<unk>") ?? 0;
  }

  static fromJSON(jsonText) {
    return new BrowserTokenizer(JSON.parse(jsonText));
  }

  getVocabSize() {
    return this.tokenToId.size;
  }

  getVocab(sortById = false) {
    if (!sortById) return Object.fromEntries(this.tokenToId);
    const out = {};
    for (const [id, tok] of [...this.idToToken.entries()].sort((a, b) => a[0] - b[0])) {
      out[tok] = id;
    }
    return out;
  }

  encode(text) {
    const normalized = nfkc(text);
    const segments = metaspaceSegments(normalized);
    const tokens = [];
    const ids = [];
    for (const seg of segments) {
      for (const tok of bpeSegment(seg, this.mergeRank)) {
        tokens.push(tok);
        ids.push(this.tokenToId.get(tok) ?? this.unkId);
      }
    }
    return { ids, tokens };
  }

  decode(ids) {
    const tokens = ids.map((id) => this.idToToken.get(id) ?? "<unk>");
    return tokens.join("").replace(/▁/g, " ");
  }
}

export async function loadTokenizer(url) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to load tokenizer (${res.status}): ${url}`);
  }
  const json = await res.text();
  return BrowserTokenizer.fromJSON(json);
}

export function tokenizerFromJSON(jsonText) {
  return BrowserTokenizer.fromJSON(jsonText);
}

export function encodeSpans(tokenizer, text) {
  const { ids, tokens } = tokenizer.encode(text);
  return ids.map((id, i) => ({ id, text: tokens[i] ?? `<${id}>` }));
}

export function checkRoundtrip(tokenizer, text) {
  const { ids } = tokenizer.encode(text);
  const decoded = tokenizer.decode(ids);
  return {
    ok: visibleNonWs(decoded) === visibleNonWs(text),
    decoded,
    tokenCount: ids.length,
  };
}
