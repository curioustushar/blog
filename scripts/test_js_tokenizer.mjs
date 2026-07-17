import fs from "fs";
import { createRequire } from "module";
import { pathToFileURL } from "url";

// Load ESM in Node for validation
const mod = await import("../static/js/bpe/hf-tokenizer.js");
const { BrowserTokenizer, checkRoundtrip } = mod;

const t = BrowserTokenizer.fromJSON(fs.readFileSync("static/bpe/tokenizer.json", "utf8"));
const rt = checkRoundtrip(t, "India's population is 1,428,627,663.");
console.log("roundtrip", rt.ok);

for (const s of ["hello world", "a\nb", "a  b"]) {
  const { tokens } = t.encode(s);
  console.log(JSON.stringify(s), tokens);
}

const full = fs.readFileSync("static/bpe/corpus/en.faithful.txt", "utf8");
console.log("en tokens", t.encode(full).ids.length, "want 116038");
