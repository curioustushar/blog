---
title: "Training Data Execution System: From Documents to Auditable Checkpoints"
slug: "training-data-execution-system"
date: 2026-08-07T08:30:00-05:00
categories: ["machine-learning", "mlops"]
tags: ["training", "data-pipeline", "reproducibility", "infrastructure", "mlops"]
author: Tushar Gupta
description: "A complete Training Data Execution System — immutable shards, ledgers, OPUS governance, checkpoint/resume/replay/fork — with auto-generated evidence and pytest invariants."
---

<div class="post-summary">

Every production ML training run must answer three questions: **What data did the model see?** **Why was it selected?** **Can this be reproduced?** This post documents the design and implementation of a complete Training Data Execution System (TDES) — from raw documents through tokenization, packing, training, checkpointing, crash recovery, and audit — with every stage producing verifiable artifacts.

<div class="plan-status" role="status" aria-label="Implementation status">
  <span class="status-badge status-ok">Demo implemented</span>
  <span class="status-badge status-ok">Evidence generated</span>
  <span class="status-badge status-ok">5 tests passing</span>
  <span class="status-badge status-pending">PyTorch training pending</span>
</div>

</div>

<nav class="post-toc" aria-label="Table of contents">

**On this page**

| | |
|---|---|
| **Overview** | [Goal](#goal) · [Architecture](#architecture) · [Pipeline](#pipeline) |
| **Data Foundation** | [Immutable shards](#immutable-shards) · [Manifests](#manifests) · [Evaluation firewall](#evaluation-firewall) |
| **Training Flow** | [Mixture scheduler](#mixture-scheduler) · [Packing](#packing) · [OPUS governance](#opus-governance) |
| **Observability** | [Consumption ledger](#consumption-ledger) · [Learning ledger](#learning-ledger) |
| **Reliability** | [Checkpoints](#checkpoints) · [Crash recovery](#crash-recovery) · [Replay](#replay) · [Audit](#audit) |
| **Implementation** | [Repository structure](#repository-structure) · [Evidence generation](#evidence-generation) |

</nav>

---

## Goal

Build a **small but complete** training data execution system that proves:

1. **Immutability** — tokenized shards are frozen; changing one token invalidates the hash
2. **Traceability** — every consumed batch maps to source documents
3. **Reproducibility** — crash → resume without skipping or repeating batches
4. **Auditability** — reconstruct complete training history from artifacts
5. **Governance** — OPUS-style acceptance/rejection/deferral with rationale

The objective is **not scale**. The objective is correctness.

This system resembles miniature versions of infrastructure at OpenAI, Anthropic, DeepMind, or Meta — compact enough to run on a laptop, rigorous enough to pass production audits.

---

## Architecture

```
┌─────────────┐
│  Documents  │  Raw corpus (Wikipedia, books, code, math)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Tokenizer  │  Frozen (hash verified on every run)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Shards    │  Immutable tokenized shards + content hash
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Manifests  │  Shard metadata (hash, count, split, source)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Eval Filter │  Block evaluation shards from training
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Mixture   │  Curriculum scheduler (lanes, weights, floors)
│  Scheduler  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Packing   │  Sequence packing (masks, position IDs)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    OPUS     │  Accept / Reject / Defer / Protected override
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Training   │  Tiny Transformer (consumption ledger active)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Checkpoint  │  Model + optimizer + ledger offset + RNG
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Crash     │  Simulated failure
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Resume    │  Next batch == expected batch (no skip/repeat)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Replay    │  Reconstruct historical batches (hash match)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    Audit    │  Reconstruct training history from artifacts
└─────────────┘
```

---

## Pipeline

### Stage 1: Documents

Small corpus with diverse types:

| Source | Split | Count | Purpose |
|--------|-------|------:|---------|
| Wikipedia | train | 1,000 | General knowledge |
| Wikipedia | eval | 100 | Evaluation firewall test |
| Books | train | 500 | Long-form text |
| Code | train | 300 | Programming |
| Math | train | 200 | Reasoning |

Every document records:
- `doc_id` (UUID)
- `source` (wikipedia / books / code / math)
- `split` (train / eval / validation)
- `text`
- `metadata` (language, license, etc.)

---

## Immutable Shards

**Definition:** A shard is a batch of tokenized documents with a frozen content hash.

```python
@dataclass
class Shard:
    shard_id: str
    token_ids: np.ndarray        # shape: (N,)
    document_ids: List[str]      # maps tokens → source docs
    offsets: np.ndarray          # document boundaries
    content_hash: str            # SHA256 of (token_ids, doc_ids, offsets)
    created_at: datetime
```

**Immutability guarantee:**

Changing **any** token invalidates the hash → shard rejected on validation.

**Why this matters:**

Production runs at Meta/Anthropic/OpenAI cannot tolerate silent data corruption. Shards are write-once, content-addressed, tamper-evident.

---

## Manifests

Every shard gets a manifest.

```python
@dataclass
class Manifest:
    shard_id: str
    content_hash: str
    tokenizer_hash: str          # ties shard to frozen tokenizer
    token_count: int
    document_count: int
    split: str                   # train / eval / validation
    source: str                  # wikipedia / books / code / math
    created_at: datetime
    version: str
```

**Validation before training:**

```python
def validate_manifest(shard: Shard, manifest: Manifest) -> bool:
    # 1. Recompute content hash
    computed = hash_shard(shard)
    
    # 2. Compare against manifest
    if computed != manifest.content_hash:
        log.error(f"Hash mismatch: {shard.shard_id}")
        return False
    
    # 3. Verify tokenizer
    if get_tokenizer_hash() != manifest.tokenizer_hash:
        log.error("Tokenizer changed since shard creation")
        return False
    
    return True
```

Logs:

```
[PASS] tokenizer_hash_verified
[PASS] shard_content_hash_verified
```

---

## Evaluation Firewall

**Rule:** Evaluation and validation shards must **never** enter training batches.

**Implementation:**

```python
def filter_training_shards(manifests: List[Manifest]) -> List[Manifest]:
    allowed = []
    blocked = []
    
    for m in manifests:
        if m.split == "train":
            allowed.append(m)
        else:
            blocked.append(m)
            log.info(f"[BLOCK] eval shard: {m.shard_id}")
    
    log.info(f"[PASS] eval_shard_blocked: {len(blocked)} shards")
    return allowed
```

**Evidence:**

```json
{
  "requirement": "evaluation_firewall",
  "status": "PASS",
  "blocked_shards": ["shard_eval_001", "shard_eval_002"],
  "evidence_file": "manifests/blocked_shards.json"
}
```

---

## Mixture Scheduler

**Curriculum stages** with lane weights and protected floors.

```python
@dataclass
class MixtureStage:
    stage_id: int
    token_range: Tuple[int, int]     # (start, end)
    lane_weights: Dict[str, float]   # {"code": 0.3, "books": 0.2, ...}
    protected_floors: Dict[str, float]  # {"math": 0.05}
```

**Example curriculum:**

| Stage | Tokens | Code | Books | Math | Chat |
|-------|-------:|-----:|------:|-----:|-----:|
| 1 | 0–100K | 10% | 50% | 20% | 20% |
| 2 | 100K–300K | 30% | 20% | 30% | 20% |
| 3 | 300K+ | 25% | 25% | 25% | 25% |

**Protected floors:**

Math lane **≥5%** regardless of OPUS score — prevents selector starvation (same principle as the 2.5T mixture plan's `Indic ≥12%`, `Agentic ≥2%` floors).

**Mixture compliance check:**

```python
planned_weights = {"code": 0.30, "books": 0.20, ...}
actual_weights = measure_actual_proportions(consumed_batches)

for lane, planned in planned_weights.items():
    actual = actual_weights[lane]
    delta = abs(actual - planned)
    
    if delta > 0.05:  # tolerance
        log.warning(f"Mixture drift: {lane} planned={planned:.2%} actual={actual:.2%}")
    else:
        log.info(f"[PASS] mixture_compliance: {lane}")
```

---

## Packing

**Challenge:** Sequences have variable length. Batches must have fixed shape.

**Solution:** Pack multiple sequences into one training example with correct masks.

```python
@dataclass
class PackedBatch:
    input_ids: torch.Tensor       # (B, L)
    attention_mask: torch.Tensor  # (B, L, L) causal mask with boundaries
    loss_mask: torch.Tensor       # (B, L) — which tokens contribute to loss
    position_ids: torch.Tensor    # (B, L) — resets at document boundaries
    sequence_ids: List[List[str]] # maps positions → source sequences
```

**Packing types:**

1. **Plain text:** concat docs with separator token
2. **Instruction tuning:** prompt loss-masked, completion loss-bearing
3. **Multi-document:** separate position IDs per doc
4. **Chat:** user/assistant turn boundaries

**Correctness invariants (tested):**

- Attention mask is causal within docs, blocked across docs
- Loss mask == 1 only for model-generated tokens (not prompts/contexts)
- Position IDs reset at every document boundary
- No tokens from eval shards in any packed batch

**Packing utilization:**

```python
total_tokens = batch_size * seq_len
padding_tokens = (loss_mask == 0).sum()
useful_tokens = total_tokens - padding_tokens

packing_utilization = useful_tokens / total_tokens
# Target: >0.90
```

---

## OPUS Governance

**Miniature data selector** — each candidate sample receives a decision:

| Decision | Meaning | Example rationale |
|----------|---------|-------------------|
| **Accept** | Include in training | High quality, on-topic |
| **Reject** | Exclude permanently | Duplicate, toxic, off-topic |
| **Defer** | Revisit later | Borderline quality |
| **Protected** | Floor override | Math lane below 5% — force accept |

**Decision ledger:**

```json
{
  "sample_id": "doc_12345",
  "decision": "accept",
  "score": 0.87,
  "rationale": "high_quality_code",
  "protected_floor": false,
  "timestamp": "2026-08-07T10:23:45Z"
}
```

**Protected floor override:**

```python
if lane == "math" and actual_proportion < floor:
    decision = "protected_override"
    rationale = "floor_enforcement"
```

**Evidence:**

OPUS audit trail must record **every decision type** during the run:

```
[PASS] opus_accept_recorded: 1247 samples
[PASS] opus_reject_recorded: 89 samples
[PASS] opus_defer_recorded: 34 samples
[PASS] opus_protected_recorded: 12 samples
```

---

## Consumption Ledger

**Purpose:** Prove exactly what entered training.

```python
@dataclass
class ConsumptionRecord:
    batch_id: str
    global_step: int
    shard_ids: List[str]
    sample_ids: List[str]
    token_spans: List[Tuple[int, int]]  # (start, end) in shard
    timestamp: datetime
```

**Storage:** Append-only log (JSONL or Parquet).

**Query example:**

```python
# What data did step 150 consume?
records = consumption_ledger.query(global_step=150)
print(records.shard_ids)
print(records.sample_ids)
```

---

## Learning Ledger

**Purpose:** Link loss to source data.

```python
@dataclass
class LearningRecord:
    batch_id: str
    global_step: int
    loss: float
    sample_ids: List[str]          # fine-grained: per-sample loss
    token_losses: Optional[List[float]]  # ultra-fine: per-token loss
    gradient_norm: float
    learning_rate: float
    checkpoint_id: Optional[str]
```

**Sample-level attribution:**

```python
for sample_id, sample_loss in zip(batch.sample_ids, per_sample_losses):
    learning_ledger.append(
        sample_id=sample_id,
        loss=sample_loss,
        step=global_step
    )
```

**Audit query:**

```python
# Which samples caused high loss?
high_loss = learning_ledger.query(loss > threshold)
print(high_loss.sample_ids)

# Trace back to source documents
docs = consumption_ledger.resolve(sample_ids=high_loss.sample_ids)
```

---

## Checkpoints

**What to save:**

```python
checkpoint = {
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "scheduler_state": scheduler.state_dict(),
    "random_states": {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    },
    "ledger_offset": consumption_ledger.current_offset(),
    "mixture_state": mixture_scheduler.state_dict(),
    "next_batch_id": next_batch_id,
    "global_step": global_step,
    "tokens_seen": tokens_seen,
}
```

**Why ledger offset matters:**

On resume, loader seeks to `ledger_offset` → continues from the **exact next sample**.

No skips. No repeats.

---

## Crash Recovery

**Scenario:**

Training crashes at step 150 (mid-run).

**Resume flow:**

1. Load checkpoint
2. Restore model, optimizer, scheduler
3. Restore RNG states
4. Seek data loader to `ledger_offset`
5. **Verify:** Next batch ID == expected batch ID
6. Continue training

**Verification:**

```python
# Before crash
expected_batch_id = "batch_151"

# After resume
actual_batch_id = next(data_loader).batch_id

assert actual_batch_id == expected_batch_id, "Resume failed"
log.info(f"[PASS] resume_next_batch_matched: {actual_batch_id}")
```

**Evidence:**

```json
{
  "requirement": "crash_recovery",
  "status": "PASS",
  "checkpoint_step": 150,
  "expected_batch": "batch_151",
  "resumed_batch": "batch_151",
  "evidence_file": "ledgers/resume_verification.json"
}
```

---

## Replay

**Goal:** Reconstruct a historical training interval **exactly**.

**Use case:**

Reproduce batch 50–100 to debug a loss spike.

**Implementation:**

```python
def replay(start_step: int, end_step: int):
    # 1. Load consumption ledger for interval
    records = consumption_ledger.query(start_step, end_step)
    
    # 2. Reconstruct batches
    replayed_batches = []
    for record in records:
        batch = reconstruct_batch(
            shard_ids=record.shard_ids,
            token_spans=record.token_spans
        )
        replayed_batches.append(batch)
    
    # 3. Compute hashes
    replayed_hashes = [hash_batch(b) for b in replayed_batches]
    
    # 4. Compare against original
    original_hashes = [r.batch_hash for r in records]
    
    assert replayed_hashes == original_hashes, "Replay failed"
    log.info(f"[PASS] replay_hash_matched: steps {start_step}-{end_step}")
```

**Why this is hard:**

- RNG states must match
- Packing order must match
- Mixture sampling must match

**Evidence:**

```json
{
  "requirement": "replay",
  "status": "PASS",
  "replayed_interval": "50-100",
  "original_hashes": ["abc123", "def456", ...],
  "replayed_hashes": ["abc123", "def456", ...],
  "match": true
}
```

---

## Audit

**Goal:** Reconstruct complete training history from artifacts.

Given a checkpoint, recover:

1. All consumed shards
2. All consumed documents
3. All batches
4. All losses
5. All OPUS decisions

**Audit report:**

```python
audit = Audit(checkpoint_id="ckpt_300")

print(audit.consumed_shards())      # → ["shard_001", "shard_002", ...]
print(audit.consumed_documents())   # → ["doc_a", "doc_b", ...]
print(audit.mixture_compliance())   # → {"code": 0.29, "books": 0.21, ...}
print(audit.opus_decisions())       # → {"accept": 1247, "reject": 89, ...}
print(audit.loss_by_source())       # → {"wikipedia": 2.3, "code": 1.8, ...}
```

**Use cases:**

- Debug training instability
- Validate mixture compliance
- Audit data usage for legal/compliance
- Reproduce specific intervals

---

## Repository Structure

```
training_data_execution_system/
├── README.md                  # Architecture, design, how to run
├── run_demo.py                # One-command execution
├── requirements.txt
├── config.py                  # Central configuration
│
├── tdes/                      # Main package
│   ├── __init__.py
│   ├── documents.py           # Document ingestion
│   ├── tokenizer.py           # Frozen tokenizer + hash
│   ├── shards.py              # Shard creation + validation
│   ├── manifests.py           # Manifest generation + verification
│   ├── firewall.py            # Evaluation blocker
│   ├── mixture.py             # Curriculum scheduler
│   ├── packing.py             # Sequence packing + masks
│   ├── opus.py                # Data selector (accept/reject/defer)
│   ├── training.py            # Tiny Transformer trainer
│   ├── ledgers.py             # Consumption + learning ledgers
│   ├── checkpoint.py          # Save/load checkpoints
│   ├── replay.py              # Historical replay
│   ├── audit.py               # Audit tooling
│   └── evidence.py            # Evidence generation
│
├── tests/                     # pytest suite
│   ├── test_shards.py
│   ├── test_manifests.py
│   ├── test_packing.py
│   ├── test_mixture.py
│   ├── test_checkpoints.py
│   ├── test_replay.py
│   └── test_evidence.py
│
├── data/                      # Raw documents
│   ├── wikipedia/
│   ├── books/
│   ├── code/
│   └── math/
│
└── submission_artifacts/      # Generated outputs
    ├── run.log
    ├── evidence.json
    ├── evidence.md
    ├── performance.json
    ├── manifests/
    ├── ledgers/
    │   ├── consumption.jsonl
    │   └── learning.jsonl
    ├── checkpoints/
    ├── replay/
    └── reports/
```

---

## Evidence Generation

**Automatic, not hardcoded.**

```python
class EvidenceCollector:
    def __init__(self):
        self.results = {}
    
    def record(self, requirement: str, status: str, artifact: str):
        self.results[requirement] = {
            "status": status,
            "artifact": artifact,
            "timestamp": datetime.now().isoformat(),
        }
    
    def generate_json(self):
        with open("submission_artifacts/evidence.json", "w") as f:
            json.dump(self.results, f, indent=2)
    
    def generate_markdown(self):
        # Auto-generate evidence.md table from results
        ...
```

**Called during run:**

```python
evidence = EvidenceCollector()

if validate_tokenizer_hash():
    evidence.record("tokenizer_integrity", "PASS", "manifests/tokenizer_hash.txt")
else:
    evidence.record("tokenizer_integrity", "FAIL", None)

if eval_shards_blocked():
    evidence.record("evaluation_firewall", "PASS", "manifests/blocked_shards.json")

# ... etc for all requirements

evidence.generate_json()
evidence.generate_markdown()
```

---

## Running the System

**Commands** (from the blog repo root):

```bash
cd training-data-execution-system
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python3 run_demo.py
.venv/bin/pytest tests/ -q
```

**Execution flow:**

1. Create corpus (2,100 documents)
2. Build and freeze tokenizer (hash verified)
3. Tokenize into immutable shards (48 shards)
4. Generate and validate manifests (48 manifests)
5. Block eval shards (5 blocked, 43 training)
6. Compile mixture schedule (planned vs actual proportions)
7. Pack batches with attention/loss/position masks
8. Train with OPUS governance + consumption/learning ledgers
9. Checkpoint, simulate crash, resume (verify next batch ID)
10. Replay steps 50–100 (packed file hashes vs ledger)
11. Fork branch from `ckpt_00050` into `consumption_fork.jsonl`
12. Audit run and generate evidence bundle

**Generated artifacts** (`submission_artifacts/`):

| File / dir | Purpose |
|------------|---------|
| `run.log` | Full execution log with required events |
| `evidence.json` / `evidence.md` | Machine- and human-readable evidence |
| `performance.json` | Measured throughput and packing utilization |
| `manifests/` | 48 shard manifests + `blocked_shards.json` |
| `ledgers/consumption.jsonl` | Per-batch consumption records |
| `ledgers/learning.jsonl` | Loss linked to sample IDs |
| `ledgers/consumption_fork.jsonl` | Fork branch ledger |
| `checkpoints/ckpt_00050.json`, `ckpt_00150.json` | Ledger-bound checkpoints |
| `replay/replay_50_100.json` | Replay hash verification |
| `reports/packed_batches/` | Packed batch JSON with masks |
| `reports/opus_decisions.json` | OPUS audit trail |

**Latest verified run** (`run_demo.py` + `pytest tests/ -q`):

| Metric | Value |
|--------|------:|
| Evidence checks | **12/12 PASS** |
| Eval shards blocked | 5 |
| Batches packed | 33 |
| Packing utilization | 87.4% |
| OPUS decisions | accept 28 · reject 60 · defer 77 · protected 5 |
| Resume match | `batch_00033` == `batch_00033` |
| Replay (steps 50–100) | hash match ✅ |
| Throughput | 8,321 tok/s (7,268 loss-bearing) |
| pytest | **5 passed** |

**Sample log output:**

```
[PASS] tokenizer_hash_verified
[PASS] eval_shard_blocked
[PASS] checkpoint_saved
[PASS] resume_next_batch_matched
[PASS] replay_hash_matched
branch forked
audit completed
[PASS] evidence_generated
Demo complete. All passed: True
```

---

## Performance Report

Measured from the training loop (not hardcoded):

```json
{
  "tokens_per_second": 8321.44,
  "loss_bearing_tokens_per_second": 7268.46,
  "packing_utilization": 0.8735,
  "padding_percentage": 12.65,
  "batches_packed": 33
}
```

---

## Design Decisions

### Why immutable shards?

Production ML training cannot tolerate silent data corruption. Content hashes make tampering evident.

### Why manifests?

Manifests decouple shard creation from validation. Training can verify data integrity without re-tokenizing.

### Why ledgers?

Ledgers provide an audit trail. Without them, you cannot prove what the model learned or reproduce a run.

### Why OPUS in a demo?

Real systems (Anthropic's Constitutional AI, OpenAI's data filters) need governance. This proves the architecture supports it.

### Why crash recovery?

Clusters fail. Preemption happens. Systems that cannot resume without data loss or duplication are not production-ready.

### Why replay?

Debugging loss spikes, validating claims, auditing compliance — all require reconstructing historical batches exactly.

---

## Evaluation Criteria (1,000 points)

| Area | Points | Evidence |
|------|-------:|----------|
| End-to-end execution | 150 | `run.log` |
| Shards, manifests, tokenizer integrity | 100 | Hash verification logs |
| Packing, masks, batch correctness | 150 | Packed batch reports + tests |
| Mixture schedule, floors, OPUS | 150 | Planned vs actual + decision ledger |
| Consumption + learning ledgers | 150 | Ledger files + query tests |
| Checkpoint, crash, resume, replay, fork | 150 | Resume verification + replay hashes |
| Evaluation firewall | 50 | Blocked shard logs |
| Throughput + packing efficiency | 50 | `performance.json` |
| Tests, evidence, documentation | 50 | pytest results + evidence bundle |
| **Total** | **1,000** | |

---

## Implementation Status

The demo is **implemented, runnable, and tested**. One command regenerates `submission_artifacts/`; `pytest tests/ -q` verifies core invariants.

### Completed

| Component | Status | Evidence |
|-----------|--------|----------|
| Corpus + frozen tokenizer | ✅ | `data/corpus.jsonl`, `data/tokenizer/tokenizer_hash.txt` |
| Immutable shards + manifests | ✅ | 48 shards, 48 manifests, hash validation |
| Evaluation firewall | ✅ | `manifests/blocked_shards.json` (5 blocked) |
| Sequence packing | ✅ | `reports/packed_batches/` — masks, position IDs, batch hashes |
| Mixture scheduler | ✅ | `reports/mixture.json` — planned vs actual |
| OPUS governance | ✅ | accept / reject / defer / protected in `opus_decisions.json` |
| Consumption ledger | ✅ | `ledgers/consumption.jsonl` |
| Learning ledger | ✅ | `ledgers/learning.jsonl` — loss per sample |
| Checkpoint + crash + resume | ✅ | `checkpoints/ckpt_00150.json`, resume batch match |
| Replay | ✅ | `replay/replay_50_100.json` — packed hashes match ledger |
| Fork | ✅ | `ledgers/consumption_fork.jsonl` (`branch_id=fork_001`) |
| Audit | ✅ | `reports/audit.json` |
| Evidence bundle | ✅ | Auto-generated `evidence.json` + `evidence.md` |
| pytest suite | ✅ | 5 tests (packing, firewall, OPUS, resume, replay) |

### Still simplified

| Component | Status | What's missing |
|-----------|--------|----------------|
| PyTorch training | Pending | Real forward/backward; pseudo-loss today |
| Manifest lineage | Partial | No cleaning-pipeline hash, contamination scan, license tier |
| Distributed loaders | N/A | Single-process demo only |

### Future work

1. Wire tiny Transformer (2-layer) with real gradients
2. Enrich manifests with Session 3–4 lineage fields
3. Expand pytest for checkpoint/resume edge cases

---

## Related Work

- [The Mixture Is the Model: A 2.5T Pretraining Plan]({{< relref "2026-07-31-the-mixture-is-the-model-2-5t-pretraining-plan.md" >}}) — Mixture design and OPUS floors
- [9 Data Cleaning Strategies Applied]({{< relref "2026-07-23-data-cleaning-strategies.md" >}}) — Upstream data quality
- [Train 40B Model Design]({{< relref "2026-07-17-train-40b-model-design.md" >}}) — D1–D7 pool layout

---

## Repository

**Code:** [training-data-execution-system/](https://github.com/curioustushar/blog/tree/master/training-data-execution-system) (in this blog repo)

```bash
cd training-data-execution-system
.venv/bin/python3 run_demo.py      # → submission_artifacts/
.venv/bin/pytest tests/ -q         # → 5 passed
```

**Status:** Demo implemented · 12/12 evidence PASS · 5 pytest tests · PyTorch training pending