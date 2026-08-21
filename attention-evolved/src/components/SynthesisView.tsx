import { MathDisplay } from "./MathDisplay";

export function SynthesisView() {
  return (
    <div className="prose prose-lg dark:prose-invert max-w-none">
      <div className="space-y-8">
        <div className="bg-muted/50 rounded-xl border border-border p-6">
          <h4 className="text-lg font-semibold mb-3">Four problems, one evolution</h4>
          <p className="text-muted-foreground mb-4">
            Nearly every mechanism on the timeline is an attempt to solve one of these:
          </p>
          <ol className="list-decimal list-inside space-y-2 text-muted-foreground">
            <li><strong>Selective retrieval</strong> — how does the model look where it needs to?</li>
            <li><strong>All-to-all interaction</strong> — how can every token reach every other token efficiently?</li>
            <li><strong>Very long sequences</strong> — how do we escape O(n²) as n grows?</li>
            <li><strong>Inference economics</strong> — how do we run enormous LLMs without drowning in KV cache and bandwidth?</li>
          </ol>
        </div>

        <div className="bg-white dark:bg-zinc-900 rounded-xl border border-border p-6">
          <h4 className="text-lg font-semibold text-accent mb-3">Before 2017: why attention existed at all</h4>
          <p className="text-muted-foreground mb-3">
            The interactive timeline starts at the Transformer (2017) because Session 2&apos;s baseline is scaled dot-product attention.
            But the idea did not appear overnight.
          </p>
          <ul className="space-y-3 text-muted-foreground text-sm">
            <li>
              <strong>2014 — Bahdanau attention.</strong> Encoder-decoder RNNs compressed the entire input into one vector.
              Bahdanau et al. let the decoder <em>look back</em> at encoder states with learned alignment weights.
              Problem solved: the information bottleneck.{" "}
              <a href="https://arxiv.org/abs/1409.0473" target="_blank" rel="noopener noreferrer" className="text-accent hover:underline">
                arXiv:1409.0473
              </a>
            </li>
            <li>
              <strong>2015 — Dot-product attention.</strong> Luong et al. replaced the alignment MLP with direct vector similarity: score(q, k) = q·k.
              This is the direct precursor to Q, K, V and the Transformer equation.{" "}
              <a href="https://arxiv.org/abs/1508.04025" target="_blank" rel="noopener noreferrer" className="text-accent hover:underline">
                arXiv:1508.04025
              </a>
            </li>
            <li>
              <strong>2017 — Self-attention + multi-head.</strong> Vaswani et al. removed the RNN entirely: Q, K, V all come from the same sequence.
              Multiple heads let different subspaces specialize. The timeline&apos;s first card bundles scaled dot-product attention with this foundation.
            </li>
          </ul>
        </div>

        <div className="bg-white dark:bg-zinc-900 rounded-xl border border-border p-6">
          <h4 className="text-lg font-semibold text-accent mb-3">One equation, four levers</h4>
          <p className="text-muted-foreground mb-4">
            Standard attention is Attention(Q, K, V) = softmax(QKᵀ / √d) V.
            Almost everything after 2017 changes one of four things:
          </p>
          <div className="bg-muted p-4 rounded-lg mb-4 overflow-x-auto">
            <MathDisplay math="O = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V" display />
          </div>
          <div className="grid md:grid-cols-2 gap-4 text-sm">
            <div className="bg-muted/60 p-3 rounded-lg">
              <p className="font-semibold text-foreground mb-1">A. Change what interacts (QKᵀ)</p>
              <p className="text-muted-foreground">Sparse Transformer, Longformer, DroPE — fewer or structured connections.</p>
            </div>
            <div className="bg-muted/60 p-3 rounded-lg">
              <p className="font-semibold text-foreground mb-1">B. Change K and V (not Q)</p>
              <p className="text-muted-foreground">MQA, GQA, MLA — share or compress what gets cached during inference.</p>
            </div>
            <div className="bg-muted/60 p-3 rounded-lg">
              <p className="font-semibold text-foreground mb-1">C. Change position, not attention math</p>
              <p className="text-muted-foreground">RoPE, ALiBi, YaRN, NTK — modify how distance enters Q and K.</p>
            </div>
            <div className="bg-muted/60 p-3 rounded-lg">
              <p className="font-semibold text-foreground mb-1">D. Change the mechanism entirely</p>
              <p className="text-muted-foreground">Linear attention, DeltaNet, Gated DeltaNet — replace softmax(QKᵀ) with recurrent or kernelized forms.</p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-zinc-900 rounded-xl border border-border p-6">
          <h4 className="text-lg font-semibold text-accent mb-3">2017: The foundation</h4>
          <p className="text-muted-foreground">
            The Transformer introduced <strong>scaled dot-product attention</strong> with <strong>sinusoidal positional encoding</strong>.
            It solved the parallelization problem of RNNs but created a new one: O(n²) compute and memory.
            At the time, sequences were short (512 tokens), so this was acceptable.
          </p>
        </div>

        <div className="bg-white dark:bg-zinc-900 rounded-xl border border-border p-6">
          <h4 className="text-lg font-semibold text-accent mb-3">2018–2019: First efficiency attempts</h4>
          <p className="text-muted-foreground">
            <strong>BERT</strong> showed learned positions worked for encoders.
            <strong>Sparse Transformer</strong> proved you could approximate full attention.
            <strong>MQA</strong> addressed KV cache size — but adoption was slow because inference at scale wasn&apos;t yet the crisis.
          </p>
        </div>

        <div className="bg-white dark:bg-zinc-900 rounded-xl border border-border p-6">
          <h4 className="text-lg font-semibold text-accent mb-3">2020: The long-context push</h4>
          <p className="text-muted-foreground">
            <strong>Longformer</strong> made long documents practical with sliding windows.
            <strong>Linear attention</strong> showed attention could be O(n) — at a quality cost.
            The field split: some wanted longer contexts, others wanted cheaper attention.
          </p>
        </div>

        <div className="bg-white dark:bg-zinc-900 rounded-xl border border-border p-6">
          <h4 className="text-lg font-semibold text-accent mb-3">2021: The position revolution</h4>
          <p className="text-muted-foreground">
            <strong>RoPE</strong> and <strong>ALiBi</strong> arrived within a day of each other (April 2021).
            Both addressed positional extrapolation with opposite philosophies.
            RoPE won adoption; ALiBi&apos;s extrapolation story was cleaner but less expressive.
            <strong>Delta rule</strong> showed linear attention could update associations, not just accumulate them.
          </p>
        </div>

        <div className="bg-white dark:bg-zinc-900 rounded-xl border border-border p-6">
          <h4 className="text-lg font-semibold text-accent mb-3">2022: Systems, not math</h4>
          <p className="text-muted-foreground">
            <strong>FlashAttention</strong> (Dao et al., May 2022) kept exact softmax attention but tiled computation to reduce GPU memory traffic.
            Not a new mechanism — a systems breakthrough. It made O(n²) affordable enough that the next bottleneck became KV cache bandwidth, not raw FLOPs.{" "}
            <a href="https://arxiv.org/abs/2205.14135" target="_blank" rel="noopener noreferrer" className="text-accent hover:underline">
              arXiv:2205.14135
            </a>
          </p>
        </div>

        <div className="bg-white dark:bg-zinc-900 rounded-xl border border-border p-6">
          <h4 className="text-lg font-semibold text-accent mb-3">2023: The inference crisis</h4>
          <p className="text-muted-foreground">
            ChatGPT made inference costs matter. <strong>GQA</strong> was the MHA/MQA compromise.
            <strong>NTK-aware scaling</strong> and <strong>YaRN</strong> extended RoPE without full retraining.
            <strong>Attention sinks</strong> revealed models dump excess attention mass into early tokens — and how to exploit that for streaming.
          </p>
        </div>

        <div className="bg-white dark:bg-zinc-900 rounded-xl border border-border p-6">
          <h4 className="text-lg font-semibold text-accent mb-3">2024–2025: Compression and recurrence return</h4>
          <p className="text-muted-foreground">
            <strong>MLA</strong> compresses KV representations themselves, not just head count.
            <strong>DeltaNet</strong> and <strong>Gated DeltaNet</strong> made recurrent-style updates parallelizable.
            <strong>DroPE</strong> asks whether precise position is needed at every distance.
            The progression: MHA → MQA → GQA → latent/compressed KV.
          </p>
        </div>

        <div className="bg-white dark:bg-zinc-900 rounded-xl border border-border p-6">
          <h4 className="text-lg font-semibold text-accent mb-3">2024–2026: Do we need attention everywhere?</h4>
          <p className="text-muted-foreground">
            State-space models (Mamba and successors) and hybrid architectures ask whether every layer needs full softmax attention.
            These are <em>alternatives</em>, not entries on this timeline — but they explain why DeltaNet-style recurrence is back.
            Modern LLMs increasingly mix branches rather than picking one winner.
          </p>
        </div>

        <div className="bg-accent/10 rounded-xl border border-accent/20 p-6">
          <h4 className="text-lg font-semibold mb-3">The pattern</h4>
          <p className="mb-4 text-muted-foreground">
            Chronological order reveals what a list hides:
          </p>
          <ol className="list-decimal list-inside space-y-2 text-muted-foreground">
            <li><strong>2014:</strong> let the model look where it needs to (Bahdanau)</li>
            <li><strong>2017:</strong> let every token look at every other token (Transformer)</li>
            <li><strong>2019–20:</strong> don&apos;t connect everything (sparse, sliding, linear)</li>
            <li><strong>2022:</strong> compute the same attention smarter (FlashAttention)</li>
            <li><strong>2023:</strong> don&apos;t store the same KV repeatedly (GQA, sinks)</li>
            <li><strong>2024+:</strong> compress what must be remembered (MLA, DeltaNet)</li>
          </ol>
          <p className="mt-4 text-muted-foreground">
            The future isn&apos;t necessarily &ldquo;better attention.&rdquo; It&apos;s the <strong>right amount of attention, over the right information, at the right cost</strong>.
            Each solution moved the bottleneck — it didn&apos;t eliminate it.
          </p>
        </div>
      </div>
    </div>
  );
}
