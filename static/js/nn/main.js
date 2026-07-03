import { initS1Activations } from "./s1-activations.js";
import { initS2Depth } from "./s2-depth.js";
import { initS3Embeddings } from "./s3-embeddings.js";
import { initS4Generalization } from "./s4-generalization.js";

function boot() {
  const map = {
    "s1-activations": initS1Activations,
    "s2-depth": initS2Depth,
    "s3-embeddings": initS3Embeddings,
    "s4-generalization": initS4Generalization,
  };
  for (const [id, fn] of Object.entries(map)) {
    const el = document.getElementById(id);
    if (el) fn(el);
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}
