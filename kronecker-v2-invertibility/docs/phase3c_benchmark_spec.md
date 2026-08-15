# Phase 3C: Full Benchmark Specification

**Status:** Designed, not yet executed  
**Prerequisite:** Phase 3B proxy must pass all tests  

---

## Objective

**Answer the central Phase 3 question:**

> Does structured byte prediction achieve comparable or better LM quality while producing meaningful reduction in output-head cost and/or actual end-to-end resource usage?

---

## Entry Criteria

### Phase 3B Must Be Complete

**All proxy tests passed:**
- ✅ Fixed-length: ≥95% exact
- ✅ Variable-length: ≥95% exact
- ✅ Scaling: all lengths work
- ✅ Codec: ≥95% success
- ✅ Transformer: trains stably
- ✅ Comparison: byte head competitive

**If Phase 3B failed:** Fix PyTorch implementation before proceeding.

---

## Experiment Structure

### Three Methods Under Test

**All share same Transformer backbone.**

#### 1. Standard Softmax (Baseline)

```python
class SoftmaxHead(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, h):
        # h: (batch, seq_len, d_model)
        return self.proj(h)  # (batch, seq_len, vocab_size)
```

**Parameters:** vocab_size × d_model

#### 2. Structured Byte Head

```python
class StructuredByteHead(nn.Module):
    def __init__(self, d_model, L):
        super().__init__()
        self.proj = nn.Linear(d_model, L * 257)
        self.L = L
    
    def forward(self, h):
        # h: (batch, seq_len, d_model)
        logits = self.proj(h)
        return logits.view(h.size(0), h.size(1), self.L, 257)
    
    def decode(self, logits, token_to_bytes_map):
        """
        logits: (batch, seq_len, L, 257)
        Returns: (batch, seq_len) token IDs
        """
        # Extract bytes
        byte_preds = logits.argmax(dim=-1)  # (batch, seq, L)
        
        # Stop at EOS, construct κ, decode
        token_ids = []
        for seq in byte_preds:
            seq_tokens = []
            for pos_bytes in seq:
                # Extract sequence up to EOS
                eos_mask = (pos_bytes == 256)
                if eos_mask.any():
                    end = eos_mask.nonzero()[0].item()
                else:
                    end = self.L
                
                byte_seq = pos_bytes[:end].clamp(0, 255)
                
                # Construct κ
                kappa = construct_kappa(byte_seq.cpu().numpy())
                
                # Decode to token ID
                decoded_bytes = algebraic_decode_raw_kronecker(
                    kappa, 256, 32, True
                )
                token_id = token_to_bytes_map.get(
                    decoded_bytes,
                    UNK_ID
                )
                seq_tokens.append(token_id)
            
            token_ids.append(seq_tokens)
        
        return torch.tensor(token_ids, device=logits.device)
```

**Parameters:** d_model × L × 257

#### 3. Continuous κ Head (Optional)

```python
class ContinuousKappaHead(nn.Module):
    def __init__(self, d_model, kappa_dim=8192):
        super().__init__()
        self.proj = nn.Linear(d_model, kappa_dim)
    
    def forward(self, h):
        return self.proj(h)
    
    def decode(self, kappa_pred, sparsity_k=32):
        """
        kappa_pred: (batch, seq_len, kappa_dim)
        Returns: (batch, seq_len) token IDs
        """
        # Top-k sparsification
        values, indices = kappa_pred.topk(sparsity_k, dim=-1)
        
        # Decode each
        token_ids = []
        for seq in kappa_pred:
            seq_tokens = []
            for kappa in seq:
                # Sparsify
                sparse_kappa = torch.zeros_like(kappa)
                topk_vals, topk_idx = kappa.topk(sparsity_k)
                sparse_kappa[topk_idx] = topk_vals
                
                # Normalize
                sparse_kappa /= sparse_kappa.norm()
                
                # Decode
                decoded_bytes = algebraic_decode_raw_kronecker(
                    sparse_kappa.cpu().numpy(),
                    256, 32, True
                )
                token_id = token_to_bytes_map.get(
                    decoded_bytes,
                    UNK_ID
                )
                seq_tokens.append(token_id)
            
            token_ids.append(seq_tokens)
        
        return torch.tensor(token_ids, device=kappa_pred.device)
```

**Parameters:** d_model × kappa_dim

**Note:** Expected to fail based on Phase 2A. Include only if time permits.

---

## Shared Transformer Backbone

```python
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, 
                 dropout, max_seq_len, output_head_type, L=None):
        super().__init__()
        
        # Shared components
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        
        self.ln_final = nn.LayerNorm(d_model)
        
        # Output head (ONLY DIFFERENCE)
        if output_head_type == 'softmax':
            self.head = SoftmaxHead(d_model, vocab_size)
        elif output_head_type == 'structured_bytes':
            assert L is not None
            self.head = StructuredByteHead(d_model, L)
        elif output_head_type == 'continuous_kappa':
            self.head = ContinuousKappaHead(d_model)
        else:
            raise ValueError(f"Unknown head: {output_head_type}")
    
    def forward(self, x, mask=None):
        # x: (batch, seq_len)
        seq_len = x.size(1)
        
        # Embed
        x = self.embed(x) + self.pos_embed(
            torch.arange(seq_len, device=x.device)
        )
        x = self.dropout(x)
        
        # Transform
        if mask is not None:
            x = self.transformer(x, src_key_padding_mask=mask)
        else:
            x = self.transformer(x)
        
        x = self.ln_final(x)
        
        # Output head
        return self.head(x)
```

**Parameters by component:**
- Embedding: vocab_size × d_model
- Transformer: ~d_model² × (n_layers × 4 × 3)
- Output head: **VARIES BY METHOD**

---

## Training Configuration

### Hyperparameters (from Phase 3A)

```yaml
model:
  d_model: 512
  n_layers: 6
  n_heads: 8
  d_ff: 2048
  dropout: 0.1
  max_seq_len: 512

training:
  optimizer: adamw
  lr: 3e-4
  weight_decay: 0.1
  warmup_steps: 2000
  max_steps: 100000
  batch_size: 128
  grad_clip: 1.0
  
data:
  dataset: wikitext-103
  vocab_size: 10000
  max_seq_len: 512
  
structured_bytes:
  L: 8
  eos_class: 256
```

### Training Loop

```python
def train_step(model, batch, optimizer, scaler):
    model.train()
    
    inputs, targets = batch
    # inputs, targets: (batch, seq_len)
    
    optimizer.zero_grad()
    
    with torch.cuda.amp.autocast():
        if isinstance(model.head, SoftmaxHead):
            logits = model(inputs)  # (B, S, V)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=PAD_ID
            )
        
        elif isinstance(model.head, StructuredByteHead):
            byte_logits = model(inputs)  # (B, S, L, 257)
            
            # Get target bytes for each token
            target_bytes = get_token_bytes(targets)  # (B, S, L)
            
            loss = F.cross_entropy(
                byte_logits.view(-1, 257),
                target_bytes.view(-1),
                ignore_index=PAD_ID
            )
        
        elif isinstance(model.head, ContinuousKappaHead):
            kappa_pred = model(inputs)  # (B, S, D)
            
            # Get target κ for each token
            target_kappa = get_token_kappa(targets)  # (B, S, D)
            
            loss = F.mse_loss(kappa_pred, target_kappa)
    
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    
    return loss.item()
```

---

## Evaluation

### Perplexity Calculation

```python
@torch.no_grad()
def evaluate(model, dataloader, token_to_bytes_map):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    for batch in dataloader:
        inputs, targets = batch
        
        if isinstance(model.head, SoftmaxHead):
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=PAD_ID,
                reduction='sum'
            )
        
        elif isinstance(model.head, StructuredByteHead):
            byte_logits = model(inputs)
            target_bytes = get_token_bytes(targets)
            loss = F.cross_entropy(
                byte_logits.view(-1, 257),
                target_bytes.view(-1),
                ignore_index=PAD_ID,
                reduction='sum'
            )
        
        # ... (similar for continuous κ)
        
        total_loss += loss.item()
        total_tokens += (targets != PAD_ID).sum().item()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity
```

### Resource Measurement

```python
def measure_resources(model, dataloader, device):
    """Measure memory and throughput."""
    
    # Memory
    torch.cuda.reset_peak_memory_stats()
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters())
    
    # Forward + backward
    for batch in dataloader:
        inputs, targets = batch.to(device)
        loss = train_step(model, (inputs, targets), optimizer, None)
        break  # One step
    
    peak_memory = torch.cuda.max_memory_allocated() / 1e9  # GB
    
    # Throughput
    model.eval()
    start = time.time()
    total_tokens = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 10:  # 10 batches
                break
            inputs, _ = batch.to(device)
            _ = model(inputs)
            total_tokens += inputs.numel()
    
    elapsed = time.time() - start
    throughput = total_tokens / elapsed  # tokens/sec
    
    return {
        'peak_memory_gb': peak_memory,
        'throughput_tokens_per_sec': throughput
    }
```

---

## Experiment Protocol

### 1. Prepare Data

```python
from datasets import load_dataset
from tokenizers import BPETokenizer

# Load WikiText-103
dataset = load_dataset('wikitext', 'wikitext-103-v1')

# Train BPE tokenizer
tokenizer = BPETokenizer()
tokenizer.train_from_dataset(
    dataset['train'],
    vocab_size=10000,
    special_tokens=['<pad>', '<unk>', '<s>', '</s>']
)

# Tokenize
def tokenize(examples):
    return tokenizer.encode_batch(examples['text'])

train_data = dataset['train'].map(tokenize, batched=True)
valid_data = dataset['validation'].map(tokenize, batched=True)
test_data = dataset['test'].map(tokenize, batched=True)

# Build token → bytes mapping
token_to_bytes = {}
for token_id in range(vocab_size):
    token_str = tokenizer.id_to_token(token_id)
    token_bytes = token_str.encode('utf-8')
    token_to_bytes[token_id] = token_bytes
```

### 2. Train All Methods

```bash
# Softmax baseline
python train.py \
  --config configs/softmax_v10k.yaml \
  --output runs/softmax_v10k_seed42 \
  --seed 42

# Structured bytes
python train.py \
  --config configs/structured_bytes_v10k.yaml \
  --output runs/structured_bytes_v10k_seed42 \
  --seed 42

# Continuous κ (optional)
python train.py \
  --config configs/continuous_kappa_v10k.yaml \
  --output runs/continuous_kappa_v10k_seed42 \
  --seed 42
```

### 3. Evaluate

```python
# Load best checkpoints
softmax_model = load_checkpoint('runs/softmax_v10k_seed42/best.pt')
byte_model = load_checkpoint('runs/structured_bytes_v10k_seed42/best.pt')

# Evaluate
softmax_ppl = evaluate(softmax_model, test_loader, None)
byte_ppl = evaluate(byte_model, test_loader, token_to_bytes)

# Measure resources
softmax_resources = measure_resources(softmax_model, train_loader, device)
byte_resources = measure_resources(byte_model, train_loader, device)

print(f"Softmax: PPL={softmax_ppl:.2f}, "
      f"Memory={softmax_resources['peak_memory_gb']:.2f}GB, "
      f"Throughput={softmax_resources['throughput_tokens_per_sec']:.0f}")

print(f"Bytes:   PPL={byte_ppl:.2f}, "
      f"Memory={byte_resources['peak_memory_gb']:.2f}GB, "
      f"Throughput={byte_resources['throughput_tokens_per_sec']:.0f}")
```

---

## Results Table Template

```markdown
# Phase 3C Results: WikiText-103 (V=10K)

## Model Configuration
- Backbone: 6 layers, 512 dim, 8 heads
- Vocabulary: 10,000 BPE tokens
- Sequence length: 512
- Training: 100K steps, 3 seeds

## Perplexity (lower is better)

| Method | Valid PPL | Test PPL | Δ vs Softmax |
|--------|-----------|----------|--------------|
| Softmax (baseline) | X.X ± Y.Y | X.X ± Y.Y | 0% |
| Structured bytes | X.X ± Y.Y | X.X ± Y.Y | ±Z% |
| Continuous κ | X.X ± Y.Y | X.X ± Y.Y | ±Z% |

## Parameters

| Method | Backbone | Output Head | Total | Δ vs Softmax |
|--------|----------|-------------|-------|--------------|
| Softmax | 25.0M | 5.1M | 30.1M | 0% |
| Structured bytes | 25.0M | 1.1M | 26.1M | -13% |
| Continuous κ | 25.0M | 4.2M | 29.2M | -3% |

## Resource Usage

| Method | Peak Memory (GB) | Throughput (tok/s) | Training Time (hr) |
|--------|------------------|--------------------|--------------------|
| Softmax | X.X | YYYY | ZZ |
| Structured bytes | X.X | YYYY | ZZ |
| Continuous κ | X.X | YYYY | ZZ |

## Interpretation

[Analysis of results]

## Verdict

[Success / Partial success / Failure based on criteria]
```

---

## Success Criteria (from Phase 3A)

### Strong Success

1. Test PPL ≤ 102% of softmax
2. Memory ≤ 90% OR Throughput ≥ 90% OR Parameters ≤ 50%

### Partial Success

1. Test PPL ≤ 105% of softmax
2. Parameter savings realized but speed/memory mixed

### Failure

1. Test PPL > 105% of softmax
2. OR Memory > baseline despite fewer params
3. OR Throughput < 70% of baseline

---

## Statistical Testing

```python
from scipy import stats

# Compare perplexities across seeds
softmax_ppls = [run1_ppl, run2_ppl, run3_ppl]
byte_ppls = [run1_ppl, run2_ppl, run3_ppl]

t_stat, p_value = stats.ttest_rel(softmax_ppls, byte_ppls)

print(f"Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")
if p_value < 0.05:
    print("Difference is statistically significant")
```

---

## Scaling Experiments (After Initial)

### V = 50K

- Larger corpus (e.g., C4 subset)
- BPE with 50K vocab
- Same training config
- Measure: PPL, parameters, resources

### V = 100K

- Even larger corpus
- 100K vocab
- Expect: parameter savings more dramatic

---

## Deliverables

1. **Trained models** (3 seeds × 2-3 methods)
2. **Results table** (perplexity, parameters, resources)
3. **Plots:**
   - Training curves (loss, PPL)
   - Resource usage comparison
   - Scaling plot (PPL vs V)
4. **Analysis document** (phase3c_results.md)
5. **Checkpoints** (for reproducibility)

---

## Timeline

- Data prep: 2-3 days
- Training (V=10K): 1-2 weeks
- Evaluation: 2-3 days
- Scaling (V=50K, 100K): 2-3 weeks
- Analysis: 1 week

**Total: 5-8 weeks**

---

## Status

**Phase 3C:** Specification complete, not yet executed  
**Entry criteria:** Phase 3B proxy must pass  
**Next:** Implement and run Phase 3B before attempting 3C
