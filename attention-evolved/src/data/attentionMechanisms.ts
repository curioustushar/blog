export type DateType = "arxiv" | "conference" | "technical-report" | "release" | "forum";

export type Category =
  | "attention-mechanism"
  | "positional-encoding"
  | "kv-cache-optimization"
  | "sparse-attention"
  | "linear-attention"
  | "recurrent-attention"
  | "positional-scaling"
  | "latent-compression";

export type ComplexityLevel = 
  | "O(1)" 
  | "O(n)" 
  | "O(n log n)" 
  | "O(n²)" 
  | "O(n·w)" 
  | "O(n√n)"
  | "O(n·d²)"
  | "O(d²)"
  | "O(L·d)"
  | "O(n·(k+w))"
  | "O(k+w)"
  | "O(n·d_c)"
  | "O(n·m)"
  | "O(n·m·d)"
  | "O(n log n) or O(n·w)"
  | "varies";

export type QualitativeRating = "excellent" | "high" | "medium" | "low" | "limited" | "none" | "varies";

import { pre2017Mechanisms } from "./pre2017Mechanisms";

export { pre2017Mechanisms };

export interface Source {
  type: "paper" | "arxiv" | "technical-report" | "blog" | "forum" | "conference";
  title: string;
  url: string;
  authors?: string[];
  date: string;
  notes?: string;
}

export interface AttentionMechanism {
  id: string;
  name: string;
  shortName?: string;
  date: string;
  dateType: DateType;
  year: number;
  category: Category;
  paper: {
    title: string;
    authors: string[];
    url: string;
  };
  problem: string;
  intuition: string;
  technicalExplanation: string;
  equation?: string;
  equationExplanation?: string;
  complexity: {
    time: ComplexityLevel;
    memory: ComplexityLevel;
    description: string;
  };
  kvCache: {
    size: QualitativeRating;
    description: string;
  };
  contextBehavior: {
    maxLength: string;
    extrapolation: QualitativeRating;
    description: string;
  };
  advantages: string[];
  disadvantages: string[];
  useCases: {
    good: string[];
    poor: string[];
  };
  visualType: "attention-matrix" | "qkv-flow" | "position-encoding" | "kv-heads" | "window" | "sparse" | "recurrent" | "compression";
  sources: Source[];
  transitionNote?: string;
}

export const attentionMechanisms: AttentionMechanism[] = [
  {
    id: "standard-attention",
    name: "Scaled Dot-Product Attention",
    shortName: "Standard Attention",
    date: "2017-06-12",
    dateType: "arxiv",
    year: 2017,
    category: "attention-mechanism",
    paper: {
      title: "Attention Is All You Need",
      authors: ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit", "Llion Jones", "Aidan N. Gomez", "Łukasz Kaiser", "Illia Polosukhin"],
      url: "https://arxiv.org/abs/1706.03762",
    },
    problem: "Recurrent networks process sequences one step at a time, creating a bottleneck for parallelization. The hidden state must compress all past information into a fixed-size vector, causing information loss on long sequences. Training is slow because each step depends on the previous.",
    intuition: "Instead of processing tokens sequentially, let every token look at every other token directly. Compute a relevance score between each pair, then use those scores to create a weighted combination of values. This gives each position direct access to the entire context.",
    technicalExplanation: "Transform input into Query (Q), Key (K), and Value (V) matrices. Compute attention scores as the dot product of Q and K, scaled by √d to prevent large values that would push softmax into saturation. Apply softmax to get weights that sum to 1, then multiply by V to get the output. Multi-head attention runs this in parallel across multiple subspaces.",
    equation: "\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V",
    equationExplanation: "Q, K, V are the query, key, and value matrices. d_k is the dimension of the keys. The softmax ensures attention weights sum to 1 for each query position.",
    complexity: {
      time: "O(n²)",
      memory: "O(n²)",
      description: "Both time and memory scale quadratically with sequence length n due to the n×n attention matrix.",
    },
    kvCache: {
      size: "high",
      description: "During autoregressive generation, the KV cache grows linearly with sequence length. For multi-head attention, cache size is 2 × n_layers × n_heads × seq_len × d_head.",
    },
    contextBehavior: {
      maxLength: "Limited by O(n²) compute",
      extrapolation: "limited",
      description: "Can theoretically handle any length but compute becomes prohibitive. With sinusoidal positions, some extrapolation is possible but degrades.",
    },
    advantages: [
      "Full global context - every token can attend to every other token",
      "Highly parallelizable - no sequential dependencies during training",
      "Captures long-range dependencies directly without information compression",
      "Interpretable attention patterns",
      "Proven to scale well with data and compute",
    ],
    disadvantages: [
      "O(n²) time and memory makes long sequences expensive",
      "No inherent notion of position - requires external positional encoding",
      "KV cache grows linearly during inference, consuming memory bandwidth",
      "Softmax forces dense attention even when sparse would suffice",
      "All-to-all communication is hardware-intensive",
    ],
    useCases: {
      good: [
        "Short to medium sequences (up to 2-4K tokens)",
        "Tasks requiring precise global reasoning",
        "Training where parallelization matters more than sequence length",
        "Encoder models where context is bounded",
      ],
      poor: [
        "Very long sequences (>8K tokens) without hardware optimization",
        "Real-time streaming where latency matters",
        "Memory-constrained environments",
        "Tasks where local context would suffice",
      ],
    },
    visualType: "attention-matrix",
    sources: [
      {
        type: "arxiv",
        title: "Attention Is All You Need",
        url: "https://arxiv.org/abs/1706.03762",
        authors: ["Vaswani et al."],
        date: "2017-06-12",
        notes: "Original Transformer paper introducing scaled dot-product attention",
      },
    ],
  },
  {
    id: "sinusoidal-positional-encoding",
    name: "Sinusoidal Positional Encoding",
    shortName: "Sinusoidal PE",
    date: "2017-06-12",
    dateType: "arxiv",
    year: 2017,
    category: "positional-encoding",
    paper: {
      title: "Attention Is All You Need",
      authors: ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit", "Llion Jones", "Aidan N. Gomez", "Łukasz Kaiser", "Illia Polosukhin"],
      url: "https://arxiv.org/abs/1706.03762",
    },
    problem: "Attention is permutation-invariant - it treats the input as a set, not a sequence. Without some notion of position, the model cannot distinguish 'the cat sat on the mat' from 'mat the on sat cat the'. Position information must be injected somehow.",
    intuition: "Use sine and cosine functions at different frequencies to create unique position encodings. Lower frequencies capture coarse position, higher frequencies capture fine position. The key insight: relative positions can be represented as linear transformations of absolute positions.",
    technicalExplanation: "For each position pos and dimension i, compute PE(pos,2i) = sin(pos/10000^(2i/d)) and PE(pos,2i+1) = cos(pos/10000^(2i/d)). These are added to token embeddings. The geometric progression of frequencies means the model can learn to attend to relative positions, since PE(pos+k) can be written as a linear function of PE(pos).",
    equation: "PE_{(pos,2i)} = \\sin\\left(\\frac{pos}{10000^{2i/d}}\\right), \\quad PE_{(pos,2i+1)} = \\cos\\left(\\frac{pos}{10000^{2i/d}}\\right)",
    equationExplanation: "pos is the position in the sequence, i is the dimension index, d is the model dimension. The 10000 base creates a geometric series of wavelengths from 2π to 10000·2π.",
    complexity: {
      time: "O(n)",
      memory: "O(n)",
      description: "Position encodings are computed once and added to embeddings. Can be precomputed.",
    },
    kvCache: {
      size: "none",
      description: "Positional encoding does not affect KV cache size - it modifies input embeddings, not attention computation.",
    },
    contextBehavior: {
      maxLength: "Theoretically unlimited",
      extrapolation: "medium",
      description: "Can compute encodings for any position, but model may not have learned to use positions beyond training length. Some length generalization is possible due to relative position structure.",
    },
    advantages: [
      "No learned parameters - works out of the box",
      "Can extrapolate to longer sequences than seen in training",
      "Relative positions are representable as linear transformations",
      "Deterministic and interpretable",
      "Efficient to compute and store",
    ],
    disadvantages: [
      "Extrapolation degrades beyond training lengths",
      "Fixed encoding may not capture task-specific position patterns",
      "Position information can be lost through layer processing",
      "Not as expressive as learned alternatives for some tasks",
    ],
    useCases: {
      good: [
        "General-purpose models without extreme length requirements",
        "When learned parameters should be minimized",
        "Machine translation and similar tasks",
        "When some length extrapolation is needed",
      ],
      poor: [
        "Tasks requiring strong length generalization",
        "When position patterns are highly task-specific",
        "Modern LLMs where RoPE has become standard",
      ],
    },
    visualType: "position-encoding",
    sources: [
      {
        type: "arxiv",
        title: "Attention Is All You Need",
        url: "https://arxiv.org/abs/1706.03762",
        authors: ["Vaswani et al."],
        date: "2017-06-12",
      },
    ],
    transitionNote: "Sinusoidal encoding solved the position problem elegantly, but the encoding is added to embeddings rather than integrated into attention. This would later be revisited with RoPE.",
  },
  {
    id: "learned-positional-embeddings",
    name: "Absolute Learned Positional Embeddings",
    shortName: "Learned Positions",
    date: "2018-10-11",
    dateType: "arxiv",
    year: 2018,
    category: "positional-encoding",
    paper: {
      title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
      authors: ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"],
      url: "https://arxiv.org/abs/1810.04805",
    },
    problem: "Sinusoidal encodings are fixed functions - they cannot adapt to the specific position patterns useful for a task. What if the model could learn the optimal position representation from data?",
    intuition: "Instead of computing positions from a formula, treat each position as a learnable embedding, just like word embeddings. The model learns during training what each position should 'mean'.",
    technicalExplanation: "Create a learnable embedding matrix of shape (max_positions, d_model). For each position p, look up embedding[p] and add it to the token embedding. This is trained end-to-end with the rest of the model. BERT used 512 learned position embeddings.",
    equation: "\\mathbf{x}_i = \\mathbf{e}_i + \\mathbf{p}_i, \\quad \\mathbf{p}_i \\in \\mathbb{R}^{d}",
    equationExplanation: "x_i is the input to the transformer, e_i is the token embedding, p_i is the learned position embedding for position i.",
    complexity: {
      time: "O(n)",
      memory: "O(L·d)",
      description: "Lookup is O(1) per position. Memory is O(max_length × d_model) for the embedding table.",
    },
    kvCache: {
      size: "none",
      description: "Does not affect KV cache - positions are added to input embeddings.",
    },
    contextBehavior: {
      maxLength: "Fixed at training",
      extrapolation: "none",
      description: "Cannot handle positions beyond max_positions - there is no embedding to look up. This is a hard limit.",
    },
    advantages: [
      "Learns task-specific position representations",
      "Simple to implement - just another embedding layer",
      "Can capture irregular position patterns if present in data",
      "No assumptions about position structure",
    ],
    disadvantages: [
      "Cannot extrapolate beyond training length at all",
      "Adds parameters proportional to max sequence length",
      "No inductive bias about relative positions",
      "Position 100 and position 101 are unrelated until trained",
    ],
    useCases: {
      good: [
        "Fixed-length inputs (e.g., BERT's 512 token limit)",
        "Classification tasks with bounded context",
        "When position patterns are irregular and task-specific",
        "Encoder-only models without generation",
      ],
      poor: [
        "Variable or very long sequences",
        "Generative models needing length flexibility",
        "When extrapolation matters",
        "Memory-constrained settings with large max_length",
      ],
    },
    visualType: "position-encoding",
    sources: [
      {
        type: "arxiv",
        title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        url: "https://arxiv.org/abs/1810.04805",
        authors: ["Devlin et al."],
        date: "2018-10-11",
        notes: "Introduced learned absolute positional embeddings for BERT",
      },
    ],
    transitionNote: "Learned positions traded extrapolation for expressiveness. This was fine for BERT's encoder-only setup, but generative models would need something better.",
  },
  {
    id: "sparse-transformer",
    name: "Sparse Attention (Sparse Transformer)",
    shortName: "Sparse Attention",
    date: "2019-04-23",
    dateType: "arxiv",
    year: 2019,
    category: "sparse-attention",
    paper: {
      title: "Generating Long Sequences with Sparse Transformers",
      authors: ["Rewon Child", "Scott Gray", "Alec Radford", "Ilya Sutskever"],
      url: "https://arxiv.org/abs/1904.10509",
    },
    problem: "Standard attention's O(n²) cost makes very long sequences (images, audio, long documents) computationally infeasible. Can we compute only the attention connections that matter most?",
    intuition: "Not all attention connections are equally important. Introduce sparse patterns where each position only attends to a subset of others. Combine local attention (nearby tokens) with strided attention (periodic global tokens) to maintain both local coherence and long-range connectivity.",
    technicalExplanation: "Replace the dense n×n attention matrix with sparse factorized patterns. Local attention connects each position to its k nearest neighbors. Strided attention connects positions that are multiples of a stride s apart. The combination achieves O(n√n) complexity while preserving connectivity between any two positions within a bounded number of attention steps.",
    equation: "\\text{Attend}(X)_i = \\text{softmax}\\left(\\frac{(X_iW^Q)(X_SW^K)^T}{\\sqrt{d}}\\right)(X_SW^V)",
    equationExplanation: "S is the sparse set of positions that position i attends to, determined by the attention pattern (local + strided).",
    complexity: {
      time: "O(n√n)",
      memory: "O(n√n)",
      description: "With factorized patterns, each position attends to O(√n) other positions instead of O(n).",
    },
    kvCache: {
      size: "medium",
      description: "KV cache can be reduced since only sparse positions need to be stored for each query.",
    },
    contextBehavior: {
      maxLength: "Greatly extended",
      extrapolation: "medium",
      description: "Can handle much longer sequences than dense attention. Used for sequences up to 16K+ tokens in the original paper.",
    },
    advantages: [
      "Reduces complexity from O(n²) to O(n√n)",
      "Enables training on much longer sequences",
      "Local + strided pattern maintains both locality and global reach",
      "Demonstrated strong results on images, text, and audio",
    ],
    disadvantages: [
      "Fixed sparse patterns may not match task requirements",
      "Implementation is more complex than dense attention",
      "Some long-range connections require multiple attention steps",
      "Requires custom CUDA kernels for efficiency",
    ],
    useCases: {
      good: [
        "Very long sequences (images as 1D sequences, long audio)",
        "Tasks where local context is primary but global context helps",
        "Training on sequences that would OOM with dense attention",
      ],
      poor: [
        "Tasks requiring arbitrary token-to-token interaction",
        "When simplicity is preferred over efficiency",
        "Short sequences where dense attention is fast enough",
      ],
    },
    visualType: "sparse",
    sources: [
      {
        type: "arxiv",
        title: "Generating Long Sequences with Sparse Transformers",
        url: "https://arxiv.org/abs/1904.10509",
        authors: ["Child et al."],
        date: "2019-04-23",
        notes: "OpenAI's Sparse Transformer paper",
      },
    ],
    transitionNote: "Sparse Transformer showed that full attention isn't always necessary - you can approximate it with structured sparsity. This opened the door to many subsequent sparse and efficient attention variants.",
  },
  {
    id: "multi-query-attention",
    name: "Multi-Query Attention",
    shortName: "MQA",
    date: "2019-09-06",
    dateType: "arxiv",
    year: 2019,
    category: "kv-cache-optimization",
    paper: {
      title: "Fast Transformer Decoding: One Write-Head is All You Need",
      authors: ["Noam Shazeer"],
      url: "https://arxiv.org/abs/1911.02150",
    },
    problem: "During autoregressive inference, the KV cache consumes memory proportional to batch_size × n_layers × n_heads × seq_len × d_head. For large models with many heads, this becomes the primary memory bottleneck. Loading the KV cache from memory is also a bandwidth bottleneck.",
    intuition: "All query heads don't need their own key and value heads. Let multiple query heads share a single key head and a single value head. This dramatically reduces the KV cache size while keeping the query expressiveness.",
    technicalExplanation: "In standard multi-head attention, each head has its own Q, K, V projections. MQA keeps multiple Q heads but collapses to a single shared K and V head. The queries are still diverse (n_heads different projections), but they all attend to the same keys and values. This reduces KV cache by a factor of n_heads.",
    equation: "\\text{MQA}(Q_i, K, V) = \\text{softmax}\\left(\\frac{Q_iK^T}{\\sqrt{d}}\\right)V \\quad \\text{(shared K, V)}",
    equationExplanation: "Each query head Q_i computes attention over the same shared K and V. The KV cache is 1/n_heads the size of MHA.",
    complexity: {
      time: "O(n²)",
      memory: "O(n²)",
      description: "Attention complexity unchanged, but KV cache memory reduced by factor of n_heads.",
    },
    kvCache: {
      size: "low",
      description: "KV cache is 1/n_heads the size of standard multi-head attention. For a 32-head model, this is a 32× reduction.",
    },
    contextBehavior: {
      maxLength: "Same as standard attention",
      extrapolation: "limited",
      description: "Context length behavior is identical to standard attention. The optimization is purely about memory efficiency.",
    },
    advantages: [
      "Dramatic reduction in KV cache memory (n_heads × smaller)",
      "Faster inference due to reduced memory bandwidth requirements",
      "Enables larger batch sizes or longer contexts in same memory",
      "Minimal quality loss for many tasks",
    ],
    disadvantages: [
      "Some quality degradation vs full multi-head attention",
      "Requires training from scratch or expensive fine-tuning",
      "Query heads may interfere when sharing KV",
      "Not a drop-in replacement for existing models",
    ],
    useCases: {
      good: [
        "Inference-optimized production models",
        "High-throughput serving with large batches",
        "Memory-constrained deployment",
        "When inference cost matters more than marginal quality",
      ],
      poor: [
        "Maximum quality is paramount",
        "Adapting existing multi-head models",
        "Research where comparisons should be apples-to-apples with MHA baselines",
      ],
    },
    visualType: "kv-heads",
    sources: [
      {
        type: "arxiv",
        title: "Fast Transformer Decoding: One Write-Head is All You Need",
        url: "https://arxiv.org/abs/1911.02150",
        authors: ["Noam Shazeer"],
        date: "2019-09-06",
        notes: "Original MQA paper by Shazeer at Google",
      },
    ],
    transitionNote: "MQA was ahead of its time - published in 2019 but not widely adopted until 2022-2023 when large-scale inference costs became critical. It trades quality for efficiency at the KV cache level.",
  },
  {
    id: "longformer",
    name: "Sliding Window Attention (Longformer)",
    shortName: "Sliding Window",
    date: "2020-04-10",
    dateType: "arxiv",
    year: 2020,
    category: "sparse-attention",
    paper: {
      title: "Longformer: The Long-Document Transformer",
      authors: ["Iz Beltagy", "Matthew E. Peters", "Arman Cohan"],
      url: "https://arxiv.org/abs/2004.05150",
    },
    problem: "Many tasks involve long documents (4K-16K+ tokens) but most information is local. Full O(n²) attention is wasteful when a token primarily needs context from nearby positions. How do we handle long documents efficiently while still allowing some global information flow?",
    intuition: "Give every token a fixed-size local window of attention. This makes the complexity linear in sequence length. For tasks that need some global context, add a small number of 'global' tokens that can attend to (and be attended by) all positions.",
    technicalExplanation: "Each token attends only to tokens within a fixed window w around it, reducing complexity to O(n·w). For specific tokens (like [CLS] for classification), enable global attention so they can aggregate information from the entire sequence. The combination handles most text with local attention while preserving global reasoning capability.",
    equation: "\\text{Attention}_i = \\text{softmax}\\left(\\frac{Q_i K_{i-w:i+w}^T}{\\sqrt{d}}\\right) V_{i-w:i+w}",
    equationExplanation: "Position i only attends to positions in the window [i-w, i+w]. Global tokens (if any) attend to all positions.",
    complexity: {
      time: "O(n·w)",
      memory: "O(n·w)",
      description: "Linear in sequence length n, with window size w as a constant factor. Typically w is 256-512.",
    },
    kvCache: {
      size: "low",
      description: "Only need to cache KV for positions within the window. Old positions can be evicted as the window slides.",
    },
    contextBehavior: {
      maxLength: "Greatly extended",
      extrapolation: "high",
      description: "Can handle sequences of 4K-16K+ tokens. Practical limit depends on global attention budget.",
    },
    advantages: [
      "Linear complexity enables very long sequences",
      "Local attention is natural for text where nearby context dominates",
      "Global tokens provide escape hatch for document-level reasoning",
      "Efficient inference - can process incrementally",
    ],
    disadvantages: [
      "Long-range dependencies require multiple layers to propagate",
      "Window size is a hyperparameter that affects different tasks differently",
      "Global tokens add complexity and must be chosen carefully",
      "Not all tasks have primarily local structure",
    ],
    useCases: {
      good: [
        "Long document understanding and QA",
        "Document classification with [CLS] as global token",
        "Summarization where local context is primary",
        "Any task where locality dominates but length is high",
      ],
      poor: [
        "Tasks requiring frequent long-range interactions",
        "When exact global attention is needed for accuracy",
        "Code or structured data where locality assumption fails",
      ],
    },
    visualType: "window",
    sources: [
      {
        type: "arxiv",
        title: "Longformer: The Long-Document Transformer",
        url: "https://arxiv.org/abs/2004.05150",
        authors: ["Beltagy et al."],
        date: "2020-04-10",
        notes: "Introduced sliding window attention with global tokens",
      },
    ],
    transitionNote: "Longformer made the locality assumption explicit: most attention is local, so make that cheap and pay for global only when needed. Sliding window would later appear in Mistral and other modern models.",
  },
  {
    id: "linear-attention",
    name: "Linear Attention",
    shortName: "Linear Attention",
    date: "2020-06-09",
    dateType: "arxiv",
    year: 2020,
    category: "linear-attention",
    paper: {
      title: "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention",
      authors: ["Angelos Katharopoulos", "Apoorv Vyas", "Nikolaos Pappas", "François Fleuret"],
      url: "https://arxiv.org/abs/2006.16236",
    },
    problem: "The softmax in attention creates a dense n×n matrix that cannot be factored. Can we change the attention kernel to allow a different computation order that avoids materializing the full matrix?",
    intuition: "Replace softmax(QK^T)V with φ(Q)(φ(K)^T V) where φ is a feature map. By associativity, compute φ(K)^T V first (a d×d matrix), then multiply by φ(Q). This is O(n·d²) instead of O(n²·d). The model becomes an RNN with a d×d state matrix.",
    technicalExplanation: "Softmax attention: softmax(QK^T)V requires computing the n×n attention matrix. Linear attention uses a kernel φ(·) such that attention can be rewritten as φ(Q)(φ(K)^T V). The key insight is that (φ(K)^T V) can be computed incrementally as a cumulative sum, giving the model a recurrent form. Common choices for φ include elu(x)+1 or the identity.",
    equation: "\\text{LinearAttn}(Q,K,V) = \\phi(Q)(\\phi(K)^TV), \\quad S_t = S_{t-1} + \\phi(k_t)v_t^T",
    equationExplanation: "φ is a feature map. The state S accumulates outer products of keys and values. For autoregressive generation, this gives O(1) computation per step.",
    complexity: {
      time: "O(n·d²)",
      memory: "O(d²)",
      description: "Time is linear in sequence length. Memory is constant (the d×d state matrix) for recurrent inference.",
    },
    kvCache: {
      size: "low",
      description: "No traditional KV cache. Instead, maintain a d×d state matrix that gets updated with each new token.",
    },
    contextBehavior: {
      maxLength: "Unlimited in principle",
      extrapolation: "high",
      description: "Can process arbitrarily long sequences with constant memory per step. In practice, information decays.",
    },
    advantages: [
      "Linear complexity in sequence length",
      "Constant memory for inference (no growing KV cache)",
      "Recurrent form enables efficient streaming inference",
      "Can handle very long sequences",
    ],
    disadvantages: [
      "Quality gap vs softmax attention, especially for tasks needing precise retrieval",
      "The d×d state cannot perfectly compress all context information",
      "Feature map choice significantly affects quality",
      "Less interpretable than explicit attention weights",
    ],
    useCases: {
      good: [
        "Very long sequences where O(n²) is prohibitive",
        "Streaming/online inference scenarios",
        "Tasks where approximate attention suffices",
        "Memory-constrained environments",
      ],
      poor: [
        "Tasks requiring precise key-value lookup",
        "When softmax attention quality is critical",
        "Short sequences where O(n²) is acceptable",
        "In-context learning that depends on exact retrieval",
      ],
    },
    visualType: "recurrent",
    sources: [
      {
        type: "arxiv",
        title: "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention",
        url: "https://arxiv.org/abs/2006.16236",
        authors: ["Katharopoulos et al."],
        date: "2020-06-09",
        notes: "Showed that linear attention creates an implicit RNN",
      },
    ],
    transitionNote: "Linear attention made explicit what many suspected: attention can be viewed as accumulating information into a state. This opened the door to state-space and recurrent alternatives.",
  },
  {
    id: "delta-rule",
    name: "Delta Rule (Fast Weight Programmers)",
    shortName: "Delta Rule",
    date: "2021-02-18",
    dateType: "arxiv",
    year: 2021,
    category: "recurrent-attention",
    paper: {
      title: "Linear Transformers Are Secretly Fast Weight Programmers",
      authors: ["Imanol Schlag", "Kazuki Irie", "Jürgen Schmidhuber"],
      url: "https://arxiv.org/abs/2102.11174",
    },
    problem: "Linear attention accumulates key-value associations additively: S += k⊗v. If the same key appears twice with different values, both associations pile up. There's no mechanism to update or overwrite old associations. This limits the model's ability to track changing state.",
    intuition: "Borrow from classical neural network learning: use the delta rule. Before adding a new association, remove what the model currently predicts for that key. This allows 'overwriting' rather than just 'accumulating'.",
    technicalExplanation: "Linear attention: S_t = S_{t-1} + k_t ⊗ v_t. Delta rule modification: S_t = S_{t-1} + k_t ⊗ (v_t - S_{t-1}k_t). The term S_{t-1}k_t is what the current state predicts for key k_t. Subtracting it before adding the new value creates an update rather than accumulation. This is equivalent to one step of gradient descent on an associative memory.",
    equation: "S_t = S_{t-1} + k_t \\otimes (v_t - S_{t-1}k_t)",
    equationExplanation: "S_{t-1}k_t retrieves the old value for key k_t. (v_t - S_{t-1}k_t) is the delta. This enables overwriting associations.",
    complexity: {
      time: "O(n·d²)",
      memory: "O(d²)",
      description: "Same complexity as linear attention - linear in n with a d×d state matrix.",
    },
    kvCache: {
      size: "low",
      description: "Uses a d×d state matrix instead of traditional KV cache.",
    },
    contextBehavior: {
      maxLength: "Unlimited",
      extrapolation: "high",
      description: "Can process arbitrarily long sequences. Better at tracking state changes than pure linear attention.",
    },
    advantages: [
      "Can update/overwrite associations, not just accumulate",
      "Better tracks changing state over long sequences",
      "Same computational complexity as linear attention",
      "Principled connection to learning and memory",
    ],
    disadvantages: [
      "Still uses a fixed-size d×d state matrix",
      "Subtraction can cause numerical instability",
      "More complex than simple linear attention",
      "Quality still below softmax for retrieval tasks",
    ],
    useCases: {
      good: [
        "Tasks with state that changes over time",
        "Long sequences where state tracking matters",
        "Algorithmic tasks with variable assignments",
        "When linear attention quality is insufficient",
      ],
      poor: [
        "Tasks requiring precise one-shot retrieval",
        "When softmax attention quality is required",
        "Simple accumulation tasks where basic linear attention suffices",
      ],
    },
    visualType: "recurrent",
    sources: [
      {
        type: "arxiv",
        title: "Linear Transformers Are Secretly Fast Weight Programmers",
        url: "https://arxiv.org/abs/2102.11174",
        authors: ["Schlag et al."],
        date: "2021-02-18",
        notes: "Introduced delta rule for linear attention",
      },
    ],
    transitionNote: "The delta rule showed that linear attention's memory could be made smarter. This sparked interest in better recurrent update rules, eventually leading to DeltaNet and Gated DeltaNet.",
  },
  {
    id: "rope",
    name: "Rotary Position Embedding",
    shortName: "RoPE",
    date: "2021-04-20",
    dateType: "arxiv",
    year: 2021,
    category: "positional-encoding",
    paper: {
      title: "RoFormer: Enhanced Transformer with Rotary Position Embedding",
      authors: ["Jianlin Su", "Yu Lu", "Shengfeng Pan", "Ahmed Murtadha", "Bo Wen", "Yunfeng Liu"],
      url: "https://arxiv.org/abs/2104.09864",
    },
    problem: "Additive positional encodings (sinusoidal or learned) inject position at the input, but this information can degrade through layers. Can we inject position information directly into the attention computation in a way that naturally encodes relative positions?",
    intuition: "Rotate query and key vectors by an angle proportional to their position. When computing Q·K, the rotation angles combine to encode the relative position (m-n). This is like encoding position in the 'phase' of a complex number.",
    technicalExplanation: "Group dimensions into pairs and treat each pair as a 2D vector. Rotate each pair by θ·position, where θ is a frequency that varies across pairs (similar to sinusoidal encoding). When computing q_m·k_n, the rotation encodes (m-n) through trigonometric identities. The dot product only depends on relative position, not absolute.",
    equation: "q_m = R_\\theta^m q, \\quad k_n = R_\\theta^n k, \\quad q_m \\cdot k_n = \\text{depends on } (m-n)",
    equationExplanation: "R_θ is a rotation matrix. Rotating Q by m·θ and K by n·θ makes their dot product depend on the relative position (m-n).",
    complexity: {
      time: "O(n)",
      memory: "O(1)",
      description: "Rotations are applied once per position. No additional memory beyond the rotated vectors.",
    },
    kvCache: {
      size: "none",
      description: "RoPE modifies Q and K directly. No effect on KV cache size.",
    },
    contextBehavior: {
      maxLength: "Training length typically",
      extrapolation: "limited",
      description: "Positions beyond training length have not been seen, so extrapolation is limited without techniques like NTK-aware scaling or YaRN.",
    },
    advantages: [
      "Directly encodes relative position in attention",
      "No added parameters",
      "Compatible with KV caching (rotate K once when caching)",
      "Flexible sequence lengths within training range",
      "Clean mathematical formulation",
    ],
    disadvantages: [
      "Limited extrapolation beyond training length",
      "Requires explicit extension techniques for longer contexts",
      "Complex to implement correctly",
      "Each dimension pair uses a different frequency",
    ],
    useCases: {
      good: [
        "Modern LLMs (LLaMA, Mistral, most open models)",
        "Tasks within training context length",
        "When relative position matters more than absolute",
        "Foundation for position interpolation techniques",
      ],
      poor: [
        "Extrapolation without extension techniques",
        "Tasks where absolute position matters specifically",
        "When simplicity is paramount",
      ],
    },
    visualType: "position-encoding",
    sources: [
      {
        type: "arxiv",
        title: "RoFormer: Enhanced Transformer with Rotary Position Embedding",
        url: "https://arxiv.org/abs/2104.09864",
        authors: ["Su et al."],
        date: "2021-04-20",
        notes: "Original RoPE paper",
      },
    ],
    transitionNote: "RoPE became the dominant positional encoding for LLMs by 2023. Its elegance and compatibility with KV caching made it the foundation that later scaling techniques (NTK, YaRN) would extend.",
  },
  {
    id: "alibi",
    name: "Attention with Linear Biases",
    shortName: "ALiBi",
    date: "2021-04-21",
    dateType: "arxiv",
    year: 2021,
    category: "positional-encoding",
    paper: {
      title: "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation",
      authors: ["Ofir Press", "Noah A. Smith", "Mike Lewis"],
      url: "https://arxiv.org/abs/2108.12409",
    },
    problem: "Existing position methods (sinusoidal, learned, RoPE) don't extrapolate well to sequences longer than training. Can we design a position method that naturally generalizes to longer contexts without retraining?",
    intuition: "Don't embed positions - just add a penalty to attention scores based on distance. The further apart two tokens are, the harder it is to attend. This creates a soft local bias that naturally extrapolates because the penalty function has no learned parameters.",
    technicalExplanation: "Add -m·|i-j| to the attention logits before softmax, where m is a head-specific slope. Different heads have different slopes (typically geometric sequence like 1/2, 1/4, 1/8...), allowing some heads to focus locally and others more globally. No position embeddings at all - just this bias term.",
    equation: "\\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d}} - m \\cdot |i-j|\\right)V",
    equationExplanation: "m is a per-head slope. The penalty -m·|i-j| discounts distant tokens. Different heads have different slopes.",
    complexity: {
      time: "O(n²)",
      memory: "O(n²)",
      description: "Same complexity as standard attention - just adds a bias matrix to attention scores.",
    },
    kvCache: {
      size: "high",
      description: "Standard KV cache size - ALiBi doesn't affect caching, only attention computation.",
    },
    contextBehavior: {
      maxLength: "Far beyond training",
      extrapolation: "excellent",
      description: "Designed for extrapolation. Models trained on 1K tokens can perform well on 2K-4K+ tokens.",
    },
    advantages: [
      "Excellent length extrapolation without retraining",
      "No learned parameters for position",
      "Simple to implement - just add a bias",
      "Different heads naturally specialize in different ranges",
    ],
    disadvantages: [
      "Implicit assumption that nearby = more relevant (may not always hold)",
      "Cannot represent absolute position if needed",
      "At very long distances, attention becomes negligible",
      "Less expressive than learned methods",
    ],
    useCases: {
      good: [
        "Models that need to generalize to longer contexts",
        "When training data has shorter contexts than deployment",
        "Tasks where locality assumption is reasonable",
        "Inference-time length flexibility",
      ],
      poor: [
        "Tasks where distant tokens are equally important",
        "When absolute position matters",
        "Tasks requiring uniform attention over long ranges",
      ],
    },
    visualType: "attention-matrix",
    sources: [
      {
        type: "arxiv",
        title: "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation",
        url: "https://arxiv.org/abs/2108.12409",
        authors: ["Press et al."],
        date: "2021-04-21",
        notes: "Original ALiBi paper. Note: arXiv date shown is from later update; original submission was April 2021",
      },
    ],
    transitionNote: "ALiBi showed that extrapolation was possible with the right inductive bias. However, the community eventually preferred RoPE + scaling techniques, finding them more expressive.",
  },
  {
    id: "grouped-query-attention",
    name: "Grouped Query Attention",
    shortName: "GQA",
    date: "2023-05-22",
    dateType: "arxiv",
    year: 2023,
    category: "kv-cache-optimization",
    paper: {
      title: "GQA: Training Generalized Multi-Query Attention from Multi-Head Checkpoints",
      authors: ["Joshua Ainslie", "James Lee-Thorp", "Michiel de Jong", "Yury Zemlyanskiy", "Federico Lebrón", "Sumit Sanghai"],
      url: "https://arxiv.org/abs/2305.13245",
    },
    problem: "MQA (1 KV head) reduces cache dramatically but can hurt quality. Full MHA (n_heads KV heads) maintains quality but has large cache. Is there a middle ground?",
    intuition: "Instead of all query heads sharing 1 KV head (MQA) or each having its own (MHA), group query heads and give each group its own KV head. This interpolates between MQA and MHA.",
    technicalExplanation: "Divide query heads into G groups. Each group shares one KV head. With n_heads=32 and G=8 groups, you have 8 KV heads instead of 32 (MHA) or 1 (MQA). The KV cache is G/n_heads the size of MHA. Quality is typically between MHA and MQA. Crucially, GQA checkpoints can be obtained by fine-tuning existing MHA models.",
    equation: "\\text{GQA}: \\text{group } i \\text{ shares } K_i, V_i \\quad (G \\text{ groups total})",
    equationExplanation: "Query heads are grouped; each group shares its K and V projections. G=1 is MQA, G=n_heads is MHA.",
    complexity: {
      time: "O(n²)",
      memory: "O(n²)",
      description: "Attention complexity unchanged. KV cache reduced by factor of n_heads/G.",
    },
    kvCache: {
      size: "medium",
      description: "KV cache size is G/n_heads of MHA. For 8 groups with 32 heads, cache is 1/4 of MHA.",
    },
    contextBehavior: {
      maxLength: "Same as base attention",
      extrapolation: "limited",
      description: "Context behavior inherits from base attention mechanism. GQA is purely a KV optimization.",
    },
    advantages: [
      "Tunable quality-efficiency tradeoff",
      "Can convert existing MHA models via fine-tuning (uptraining)",
      "Less quality loss than MQA for similar speedup",
      "Inference speedup from reduced KV cache bandwidth",
    ],
    disadvantages: [
      "Still requires some fine-tuning to convert MHA models",
      "G is another hyperparameter to tune",
      "Less aggressive reduction than MQA",
      "Added complexity vs simple MHA or MQA",
    ],
    useCases: {
      good: [
        "Production LLMs (LLaMA 2 70B uses GQA)",
        "When MQA quality loss is too high",
        "Converting existing MHA checkpoints for efficient inference",
        "Balancing quality and throughput",
      ],
      poor: [
        "When maximum quality is required (use MHA)",
        "When maximum efficiency is required (use MQA)",
        "When fine-tuning budget is zero",
      ],
    },
    visualType: "kv-heads",
    sources: [
      {
        type: "arxiv",
        title: "GQA: Training Generalized Multi-Query Attention from Multi-Head Checkpoints",
        url: "https://arxiv.org/abs/2305.13245",
        authors: ["Ainslie et al."],
        date: "2023-05-22",
        notes: "Google's GQA paper, used in LLaMA 2",
      },
    ],
    transitionNote: "GQA found the middle ground between MQA and MHA. Its adoption in LLaMA 2 made it the standard choice for production LLMs by late 2023.",
  },
  {
    id: "ntk-aware-scaling",
    name: "NTK-aware Scaled RoPE",
    shortName: "NTK-aware Scaling",
    date: "2023-06-28",
    dateType: "forum",
    year: 2023,
    category: "positional-scaling",
    paper: {
      title: "NTK-Aware Scaled RoPE allows LLaMA models to have extended (8k+) context size without any fine-tuning",
      authors: ["bloc97 (pseudonymous)"],
      url: "https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/",
    },
    problem: "RoPE doesn't extrapolate well beyond training length. Simple position interpolation (scaling positions by training_length/target_length) destroys high-frequency information. How can we extend context without losing important position signals?",
    intuition: "Position interpolation uniformly scales all frequencies, but not all frequencies are equally important. Low frequencies carry coarse position info and can be interpolated. High frequencies carry fine-grained info and should be preserved. Adjust the RoPE base frequency (from 10000) to achieve a 'Neural Tangent Kernel'-inspired scaling.",
    technicalExplanation: "Instead of interpolating positions (which scales all frequencies), increase the RoPE base from 10000 to a higher value. This makes high frequencies slower (preserving local resolution) while allowing low frequencies to extend further. The scaling factor is computed to maintain the NTK property where high-frequency behavior is preserved.",
    equation: "\\theta'_{i} = \\theta_i \\cdot \\alpha^{d/(d-2)}, \\quad \\alpha = \\frac{\\text{target\\_len}}{\\text{train\\_len}}",
    equationExplanation: "The RoPE base is scaled by α raised to a power that preserves high-frequency information while extending low frequencies.",
    complexity: {
      time: "O(1)",
      memory: "O(1)",
      description: "Just changes the RoPE base parameter - no computational overhead.",
    },
    kvCache: {
      size: "none",
      description: "Does not affect KV cache - just changes position encoding computation.",
    },
    contextBehavior: {
      maxLength: "Extended 2-4× without fine-tuning",
      extrapolation: "high",
      description: "Allows models trained on 4K to work on 8K-16K with no fine-tuning, though quality degrades at extremes.",
    },
    advantages: [
      "Extends context without any fine-tuning",
      "Simple to implement - just change base",
      "Preserves high-frequency (local) position information",
      "Works immediately on existing RoPE models",
    ],
    disadvantages: [
      "Quality degrades at very long extensions",
      "Empirical rather than fully principled",
      "Not as good as fine-tuned extensions",
      "Finding optimal scaling requires experimentation",
    ],
    useCases: {
      good: [
        "Quick context extension without retraining",
        "Inference-time length flexibility",
        "When fine-tuning is not possible",
        "Modest extensions (2-4× training length)",
      ],
      poor: [
        "Very long contexts (10×+ training)",
        "When maximum quality is required",
        "Production systems needing predictable behavior",
      ],
    },
    visualType: "position-encoding",
    sources: [
      {
        type: "forum",
        title: "NTK-Aware Scaled RoPE allows LLaMA models to have extended (8k+) context size without any fine-tuning",
        url: "https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/",
        authors: ["bloc97"],
        date: "2023-06-28",
        notes: "Originally posted on Reddit r/LocalLLaMA. Community-driven research.",
      },
    ],
    transitionNote: "NTK-aware scaling was a breakthrough from the open-source community, showing that context extension didn't require fine-tuning. This sparked further research into YaRN and other extensions.",
  },
  {
    id: "yarn",
    name: "YaRN (Yet another RoPE extensioN)",
    shortName: "YaRN",
    date: "2023-08-31",
    dateType: "arxiv",
    year: 2023,
    category: "positional-scaling",
    paper: {
      title: "YaRN: Efficient Context Window Extension of Large Language Models",
      authors: ["Bowen Peng", "Jeffrey Quesnelle", "Honglu Fan", "Enrico Shippole"],
      url: "https://arxiv.org/abs/2309.00071",
    },
    problem: "NTK-aware scaling helps but still degrades at long extensions. Pure interpolation hurts high frequencies. How do we get the best of both approaches?",
    intuition: "Different frequency components should be treated differently. Very low frequencies can be interpolated (they only need coarse position). Very high frequencies should be extrapolated (they encode local relationships). Middle frequencies get a blend.",
    technicalExplanation: "YaRN divides RoPE dimensions into three regions based on wavelength: low frequencies are interpolated, high frequencies are extrapolated (NTK-style), and middle frequencies use a ramp that blends between the two. This 'piecewise' approach preserves local structure while extending global range. Also applies a temperature scaling to attention logits to compensate for distribution shift.",
    equation: "\\theta'_i = \\begin{cases} \\theta_i \\cdot s & \\text{if } \\lambda_i \\geq \\lambda_{\\text{high}} \\\\ \\theta_i & \\text{if } \\lambda_i \\leq \\lambda_{\\text{low}} \\\\ \\text{interp} & \\text{otherwise} \\end{cases}",
    equationExplanation: "λ is the wavelength for each dimension. High wavelengths (low freq) are scaled, low wavelengths (high freq) are kept, middle is interpolated.",
    complexity: {
      time: "O(1)",
      memory: "O(1)",
      description: "Only changes RoPE computation parameters. No additional compute or memory.",
    },
    kvCache: {
      size: "none",
      description: "Does not affect KV cache.",
    },
    contextBehavior: {
      maxLength: "Extended 4-16× with fine-tuning",
      extrapolation: "excellent",
      description: "With short fine-tuning (400-1000 steps), can extend 4K context to 64K-128K.",
    },
    advantages: [
      "State-of-the-art quality at extended lengths",
      "Minimal fine-tuning required (hundreds of steps)",
      "Principled treatment of different frequency bands",
      "Temperature scaling compensates for attention distribution shift",
    ],
    disadvantages: [
      "Requires some fine-tuning for best results",
      "More complex than simple interpolation or NTK",
      "Hyperparameters (α, β for ramp) need tuning",
      "Limited theory for why the ramp works",
    ],
    useCases: {
      good: [
        "Production long-context LLMs",
        "When fine-tuning budget is available (even small)",
        "Extending 4K models to 32K-128K",
        "When quality at extended lengths matters",
      ],
      poor: [
        "Zero-shot extension (use NTK-aware)",
        "When simplicity is paramount",
        "Extremely long extensions (>32× training)",
      ],
    },
    visualType: "position-encoding",
    sources: [
      {
        type: "arxiv",
        title: "YaRN: Efficient Context Window Extension of Large Language Models",
        url: "https://arxiv.org/abs/2309.00071",
        authors: ["Peng et al."],
        date: "2023-08-31",
        notes: "Current state-of-the-art for RoPE context extension",
      },
    ],
    transitionNote: "YaRN unified insights from interpolation and NTK scaling into a principled framework. It became the standard approach for extending context windows in 2023-2024.",
  },
  {
    id: "attention-sinks",
    name: "Attention Sinks (StreamingLLM)",
    shortName: "Attention Sinks",
    date: "2023-09-29",
    dateType: "arxiv",
    year: 2023,
    category: "attention-mechanism",
    paper: {
      title: "Efficient Streaming Language Models with Attention Sinks",
      authors: ["Guangxuan Xiao", "Yuandong Tian", "Beidi Chen", "Song Han", "Mike Lewis"],
      url: "https://arxiv.org/abs/2309.17453",
    },
    problem: "For streaming inference with a sliding window, when you drop old tokens from the KV cache, quality collapses. The model seems to need the first few positions even when they're far outside the 'relevant' window. Why?",
    intuition: "Models learn to use the first few token positions as 'attention sinks' - places to dump attention when nothing else is relevant. It's not that position 0 contains important information; the model just needs somewhere to put excess attention mass. Preserve these sink positions, and sliding window streaming works.",
    technicalExplanation: "During training, the softmax attention always includes position 0 (and perhaps a few more initial positions). The model learns to use these as default targets when a query has nothing specific to attend to. In streaming inference, keep the first k 'sink' tokens in the KV cache permanently, then use a sliding window for the rest. This prevents the attention collapse that occurs when sinks are removed.",
    equation: "\\text{Cache} = [\\text{sink tokens}] + [\\text{recent window}]",
    equationExplanation: "Keep first k sink tokens permanently. Sliding window for recent positions. Total cache size is k + window.",
    complexity: {
      time: "O(n·(k+w))",
      memory: "O(k+w)",
      description: "With k sink tokens and window w, attention is O(k+w) per token. Cache is constant size.",
    },
    kvCache: {
      size: "low",
      description: "Fixed-size cache: k sink tokens + w recent tokens. Does not grow with sequence length.",
    },
    contextBehavior: {
      maxLength: "Unlimited streaming",
      extrapolation: "high",
      description: "Can process arbitrarily long sequences with constant memory. Context is limited to sink + window.",
    },
    advantages: [
      "Enables unlimited-length streaming with fixed memory",
      "Simple modification to existing sliding window",
      "Reveals important insight about attention behavior",
      "Works with existing models without retraining",
    ],
    disadvantages: [
      "Cannot recall information outside sink + window",
      "Sink tokens consume some of the limited cache budget",
      "Only solves streaming, not full context understanding",
      "Early token positions are 'wasted' on sink functionality",
    ],
    useCases: {
      good: [
        "Streaming inference (chat, real-time processing)",
        "Fixed-memory deployment",
        "Tasks where only recent context matters",
        "Long sessions where old context can be forgotten",
      ],
      poor: [
        "Tasks requiring recall of old information",
        "Document QA where any part might be relevant",
        "When full context understanding is needed",
      ],
    },
    visualType: "attention-matrix",
    sources: [
      {
        type: "arxiv",
        title: "Efficient Streaming Language Models with Attention Sinks",
        url: "https://arxiv.org/abs/2309.17453",
        authors: ["Xiao et al."],
        date: "2023-09-29",
        notes: "Discovered the attention sink phenomenon and its fix",
      },
    ],
    transitionNote: "Attention sinks revealed that models develop unexpected behaviors during training. This insight influenced how practitioners thought about KV cache management.",
  },
  {
    id: "multi-head-latent-attention",
    name: "Multi-head Latent Attention",
    shortName: "MLA",
    date: "2024-05-06",
    dateType: "technical-report",
    year: 2024,
    category: "latent-compression",
    paper: {
      title: "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model",
      authors: ["DeepSeek-AI"],
      url: "https://arxiv.org/abs/2405.04434",
    },
    problem: "MQA/GQA reduce KV cache but still store full-dimensional K and V vectors per position. As models get larger and contexts get longer, even reduced KV caches become bandwidth bottlenecks. Can we compress the KV representations further?",
    intuition: "Instead of caching K and V directly, compress them into a much smaller latent representation. During attention, reconstruct K and V from the compressed latent. This trades some compute for much less memory bandwidth.",
    technicalExplanation: "Store a low-dimensional 'latent' vector c for each position (dimension d_c << d_model). During attention, project c back up to K and V: K = W_UK · c, V = W_UV · c. The cache stores only c, reducing memory by factor d_model/d_c. Also absorbs RoPE into the projection matrices to avoid storing rotated keys. Adds some compute but saves memory bandwidth, which is often the bottleneck.",
    equation: "c_t = W_C h_t, \\quad K_t = W_{UK} c_t, \\quad V_t = W_{UV} c_t",
    equationExplanation: "h is the hidden state. c is the compressed latent (d_c << d_model). K and V are reconstructed from c during attention.",
    complexity: {
      time: "O(n²)",
      memory: "O(n·d_c)",
      description: "Attention is still O(n²), but KV cache is compressed by factor d_model/d_c.",
    },
    kvCache: {
      size: "low",
      description: "KV cache stores compressed latents instead of full K,V. DeepSeek-V2 uses d_c=512 with d_model=5120, a 10× compression.",
    },
    contextBehavior: {
      maxLength: "Same as base attention",
      extrapolation: "limited",
      description: "Context length behavior inherits from base attention. MLA is a compression technique.",
    },
    advantages: [
      "Dramatic KV cache reduction (5-10× or more)",
      "Reduces memory bandwidth bottleneck during inference",
      "Enables larger batch sizes or longer contexts",
      "Principled compression rather than dropping heads",
    ],
    disadvantages: [
      "Added compute to reconstruct K, V from latents",
      "Compression may lose information",
      "Complex implementation",
      "Training from scratch or significant fine-tuning required",
    ],
    useCases: {
      good: [
        "Large-scale production inference",
        "Memory-bandwidth-limited scenarios",
        "Very long contexts where KV cache dominates",
        "When batch throughput matters more than latency",
      ],
      poor: [
        "Latency-critical applications (reconstruction adds compute)",
        "When maximum quality is paramount",
        "Small models where KV cache isn't the bottleneck",
      ],
    },
    visualType: "compression",
    sources: [
      {
        type: "technical-report",
        title: "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model",
        url: "https://arxiv.org/abs/2405.04434",
        authors: ["DeepSeek-AI"],
        date: "2024-05-06",
        notes: "Introduced MLA as part of DeepSeek-V2 architecture",
      },
    ],
    transitionNote: "MLA represented a shift from reducing the number of KV heads to compressing what each head stores. This is orthogonal to MQA/GQA and can be combined with them.",
  },
  {
    id: "deltanet",
    name: "DeltaNet",
    shortName: "DeltaNet",
    date: "2024-10-02",
    dateType: "arxiv",
    year: 2024,
    category: "recurrent-attention",
    paper: {
      title: "Parallelizing Linear Transformers with the Delta Rule over Sequence Length",
      authors: ["Songlin Yang", "Bailin Wang", "Yu Zhang", "Yikang Shen", "Yoon Kim"],
      url: "https://arxiv.org/abs/2406.06484",
    },
    problem: "The delta rule update S = S + k⊗(v - Sk) is recurrent - each state depends on the previous. This prevents parallelization across sequence length during training. Can we make delta-rule updates parallelizable?",
    intuition: "Derive a closed-form expression that computes the final state from all inputs in parallel, similar to how linear attention can be computed as φ(Q)(φ(K)^T V). This enables GPU-efficient training while keeping the delta-rule's ability to update associations.",
    technicalExplanation: "The delta rule creates a system of linear recurrences. By unrolling and rearranging, one can express the state computation in terms of matrix operations that can be parallelized. The key insight is that the 'retrieval' and 'write' operations can be separated into parallel-computable forms. DeltaNet achieves O(n·d²) complexity while maintaining the delta rule's update semantics.",
    equation: "S_t = S_0 \\prod_{j=1}^{t}(I - \\beta_j k_j k_j^T) + \\sum_{i=1}^{t}(v_i - S_{i-1}k_i)k_i^T \\prod_{j=i+1}^{t}(I-\\beta_j k_j k_j^T)",
    equationExplanation: "The state can be written as a product/sum that admits parallel computation via matrix operations.",
    complexity: {
      time: "O(n·d²)",
      memory: "O(d²)",
      description: "Linear in sequence length. The d×d state matrix dominates memory.",
    },
    kvCache: {
      size: "low",
      description: "Uses a d×d state matrix instead of per-position KV cache.",
    },
    contextBehavior: {
      maxLength: "Unlimited",
      extrapolation: "high",
      description: "Can process arbitrarily long sequences with constant memory during inference.",
    },
    advantages: [
      "Parallel training unlike original delta rule",
      "Better associative memory than linear attention",
      "Constant memory during inference",
      "Hardware-efficient implementation possible",
    ],
    disadvantages: [
      "Still bounded by d×d state capacity",
      "Quality gap vs softmax attention on some tasks",
      "Complex implementation",
      "Relatively new, less validated at scale",
    ],
    useCases: {
      good: [
        "Very long sequences requiring efficient training",
        "Tasks with state-tracking requirements",
        "Memory-constrained inference",
        "When linear attention quality is insufficient",
      ],
      poor: [
        "Tasks requiring precise retrieval",
        "When softmax attention quality is needed",
        "Short sequences where O(n²) is acceptable",
      ],
    },
    visualType: "recurrent",
    sources: [
      {
        type: "arxiv",
        title: "Parallelizing Linear Transformers with the Delta Rule over Sequence Length",
        url: "https://arxiv.org/abs/2406.06484",
        authors: ["Yang et al."],
        date: "2024-10-02",
        notes: "Made delta rule updates parallelizable",
      },
    ],
    transitionNote: "DeltaNet showed that recurrent updates and parallel training don't have to be mutually exclusive. This enabled practical training of delta-rule models.",
  },
  {
    id: "gated-deltanet",
    name: "Gated DeltaNet",
    shortName: "Gated DeltaNet",
    date: "2024-10-02",
    dateType: "arxiv",
    year: 2024,
    category: "recurrent-attention",
    paper: {
      title: "Parallelizing Linear Transformers with the Delta Rule over Sequence Length",
      authors: ["Songlin Yang", "Bailin Wang", "Yu Zhang", "Yikang Shen", "Yoon Kim"],
      url: "https://arxiv.org/abs/2406.06484",
    },
    problem: "DeltaNet uses a fixed update rule. Can we make the memory update data-dependent, allowing the model to decide when to write strongly vs weakly?",
    intuition: "Add learnable gates (like LSTM/GRU) that control how much to update the memory for each token. Some tokens should write strongly, others should barely affect the state.",
    technicalExplanation: "Extend DeltaNet with input-dependent gating. The update becomes S = (1-β)S + β·k⊗(v - Sk) where β is computed from the input. When β≈0, the state is preserved. When β≈1, a full delta update occurs. This allows the model to learn when to write and when to read-only.",
    equation: "S_t = (1-\\beta_t)S_{t-1} + \\beta_t k_t \\otimes (v_t - S_{t-1}k_t)",
    equationExplanation: "β_t is a learned gate. It interpolates between preserving the old state (β=0) and full delta update (β=1).",
    complexity: {
      time: "O(n·d²)",
      memory: "O(d²)",
      description: "Same complexity as DeltaNet. Gating adds minimal overhead.",
    },
    kvCache: {
      size: "low",
      description: "Uses d×d state matrix like DeltaNet.",
    },
    contextBehavior: {
      maxLength: "Unlimited",
      extrapolation: "high",
      description: "Same as DeltaNet - constant memory inference.",
    },
    advantages: [
      "Data-dependent updates improve expressiveness",
      "Can selectively preserve or update memory",
      "Maintains DeltaNet's parallelizability",
      "Better performance than ungated version",
    ],
    disadvantages: [
      "Added complexity over DeltaNet",
      "Gate parameters increase model size slightly",
      "Same d×d state capacity limit",
      "New technique, limited large-scale validation",
    ],
    useCases: {
      good: [
        "Tasks with selective memory requirements",
        "When some tokens are 'more important' than others",
        "State-tracking with variable update intensity",
        "Long sequences with sparse important events",
      ],
      poor: [
        "Tasks requiring precise global attention",
        "When simplicity is preferred",
        "Short sequences where gating overhead isn't justified",
      ],
    },
    visualType: "recurrent",
    sources: [
      {
        type: "arxiv",
        title: "Parallelizing Linear Transformers with the Delta Rule over Sequence Length",
        url: "https://arxiv.org/abs/2406.06484",
        authors: ["Yang et al."],
        date: "2024-10-02",
        notes: "Introduced alongside DeltaNet in the same paper",
      },
    ],
    transitionNote: "Gated DeltaNet brought LSTM-style gating to modern linear attention, combining the best of recurrent architectures with efficient training.",
  },
  {
    id: "drope",
    name: "DroPE (Dropped Position Embedding)",
    shortName: "DroPE",
    date: "2025-12-13",
    dateType: "arxiv",
    year: 2025,
    category: "positional-scaling",
    paper: {
      title: "A Training-Free Sub-Quadratic Cost Transformer Model Serving Framework With Hierarchically Pruned Attention",
      authors: ["Heejun Lee", "Geon Park", "Jaduk Suh", "Sung Ju Hwang"],
      url: "https://arxiv.org/abs/2412.09797",
    },
    problem: "Even with RoPE extensions like YaRN, very long contexts (128K+) remain computationally expensive. Full attention is still O(n²). Can we reduce cost while maintaining long-context ability?",
    intuition: "At very long distances, exact position might matter less than approximate position. For distant tokens, we can 'drop' fine-grained positional information while preserving coarse position, enabling more aggressive attention sparsification.",
    technicalExplanation: "DroPE applies hierarchical position encoding: nearby tokens get precise positions, distant tokens get coarser position bins. This allows a hierarchical attention pattern where local attention is precise and global attention is approximate. Combined with attention pruning, achieves sub-quadratic cost while maintaining long-context understanding.",
    equation: "\\text{RoPE}'(q, k, m, n) = \\begin{cases} \\text{RoPE}(q,k,m,n) & |m-n| \\leq w \\\\ \\text{RoPE}_{\\text{coarse}}(q,k,m,n) & |m-n| > w \\end{cases}",
    equationExplanation: "Local positions (within window w) use full RoPE. Distant positions use coarsened RoPE, enabling pruning.",
    complexity: {
      time: "O(n log n) or O(n·w)",
      memory: "O(n·w)",
      description: "Sub-quadratic through hierarchical attention and pruning. Exact complexity depends on implementation.",
    },
    kvCache: {
      size: "medium",
      description: "Can reduce effective KV cache through hierarchical structure.",
    },
    contextBehavior: {
      maxLength: "Very long (128K+)",
      extrapolation: "high",
      description: "Designed for very long contexts. Trades some precision for efficiency at distance.",
    },
    advantages: [
      "Sub-quadratic cost for long contexts",
      "Training-free - works on existing models",
      "Maintains reasonable quality at distance",
      "Enables 128K+ context practically",
    ],
    disadvantages: [
      "Loses fine-grained distant position information",
      "Approximate global attention may miss some patterns",
      "Complex implementation",
      "Very recent, limited validation",
    ],
    useCases: {
      good: [
        "Very long documents (100K+ tokens)",
        "When local precision + global awareness suffices",
        "Training-free deployment on existing models",
        "Cost-sensitive long-context inference",
      ],
      poor: [
        "Tasks requiring precise distant relationships",
        "Short sequences (overhead not justified)",
        "Maximum quality requirements",
      ],
    },
    visualType: "sparse",
    sources: [
      {
        type: "arxiv",
        title: "A Training-Free Sub-Quadratic Cost Transformer Model Serving Framework With Hierarchically Pruned Attention",
        url: "https://arxiv.org/abs/2412.09797",
        authors: ["Lee et al."],
        date: "2025-12-13",
        notes: "Introduced hierarchical position dropping for efficiency",
      },
    ],
    transitionNote: "DroPE represents the frontier of position encoding research - asking whether we need precise positions everywhere, or whether hierarchy can buy efficiency.",
  },
];

export const getByYear = (year: number): AttentionMechanism[] => {
  return attentionMechanisms.filter((m) => m.year === year);
};

export const getByCategory = (category: Category): AttentionMechanism[] => {
  return attentionMechanisms.filter((m) => m.category === category);
};

export const allMechanisms: AttentionMechanism[] = [
  ...pre2017Mechanisms,
  ...attentionMechanisms,
];

export const getById = (id: string): AttentionMechanism | undefined => {
  return allMechanisms.find((m) => m.id === id);
};

export const getAllYears = (mechanisms: AttentionMechanism[] = attentionMechanisms): number[] => {
  return [...new Set(mechanisms.map((m) => m.year))].sort();
};

export const getAllCategories = (mechanisms: AttentionMechanism[] = attentionMechanisms): Category[] => {
  return [...new Set(mechanisms.map((m) => m.category))];
};

export const categoryLabels: Record<Category, string> = {
  "attention-mechanism": "Core Attention",
  "positional-encoding": "Positional Encoding",
  "kv-cache-optimization": "KV Cache Optimization",
  "sparse-attention": "Sparse Attention",
  "linear-attention": "Linear Attention",
  "recurrent-attention": "Recurrent Attention",
  "positional-scaling": "Position Scaling",
  "latent-compression": "Latent Compression",
};

export const categoryColors: Record<Category, string> = {
  "attention-mechanism": "#3b82f6",
  "positional-encoding": "#8b5cf6",
  "kv-cache-optimization": "#10b981",
  "sparse-attention": "#f59e0b",
  "linear-attention": "#ef4444",
  "recurrent-attention": "#ec4899",
  "positional-scaling": "#6366f1",
  "latent-compression": "#14b8a6",
};
