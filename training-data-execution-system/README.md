# Training Data Execution System (TDES)

A production-grade training data pipeline demonstrating immutable shards, manifests, consumption ledgers, OPUS governance, crash recovery, and deterministic replay.

**Goal:** Prove that every byte consumed during training is traceable, reproducible, and auditable.

## Architecture

```
Documents → Tokenizer → Shards → Manifests → Eval Filter → 
Mixture Scheduler → Packing → OPUS → Training → 
Consumption Ledger → Learning Ledger → Checkpoint → 
Crash → Resume → Replay → Audit
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete demonstration
python run_demo.py
```

This will:
1. Create a small corpus
2. Build and freeze a tokenizer
3. Generate immutable tokenized shards
4. Create manifests
5. Block evaluation shards
6. Run training with mixture scheduling
7. Save checkpoint
8. Simulate crash
9. Resume training
10. Replay historical batches
11. Generate audit report
12. Produce evidence bundle

## Output

After running, check `submission_artifacts/`:

```
submission_artifacts/
├── run.log                    # Complete execution log
├── evidence.json              # Machine-readable evidence
├── evidence.md                # Human-readable evidence table
├── performance.json           # Throughput metrics
├── manifests/                 # Shard manifests
├── ledgers/
│   ├── consumption.jsonl      # What was consumed
│   └── learning.jsonl         # What was learned
├── checkpoints/               # Model checkpoints
├── replay/                    # Replay verification
└── reports/                   # Mixture compliance, OPUS decisions
```

## Key Features

### 1. Immutable Shards
- Tokenized documents with content hashes
- Changing one token invalidates the hash
- Tamper-evident

### 2. Manifests
- Metadata for every shard
- Tokenizer hash verification
- Split tracking (train/eval/validation)

### 3. Evaluation Firewall
- Eval shards blocked from training
- Logged and auditable

### 4. Mixture Scheduler
- Curriculum stages with lane weights
- Protected floors (e.g., math ≥5%)
- Planned vs actual compliance tracking

### 5. Packing
- Sequence packing with correct masks
- Attention mask (causal + boundaries)
- Loss mask (model-generated only)
- Position IDs (reset at doc boundaries)

### 6. OPUS Governance
- Accept / Reject / Defer / Protected override
- Decision rationale logged
- Audit trail for every candidate

### 7. Consumption Ledger
- Records every consumed batch
- Maps to source shards and documents
- Enables traceability

### 8. Learning Ledger
- Links loss to source data
- Sample-level or token-level attribution
- Supports audit queries

### 9. Crash Recovery
- Checkpoint includes ledger offset
- Resume without skipping or repeating batches
- Verified: next batch == expected batch

### 10. Replay
- Reconstruct historical batches
- Hash verification
- Deterministic

### 11. Audit
- Reconstruct complete training history
- From checkpoint, recover all consumed data
- Mixture compliance verification

## Design Decisions

### Why Immutable Shards?
Production systems cannot tolerate silent data corruption. Content hashes make tampering evident.

### Why Ledgers?
Without ledgers, you cannot prove what the model learned or reproduce a run. Ledgers provide the audit trail.

### Why OPUS?
Real systems need data governance. This proves the architecture supports acceptance/rejection/deferral with rationale.

### Why Crash Recovery?
Clusters fail. Preemption happens. Systems that cannot resume are not production-ready.

### Why Replay?
Debugging loss spikes, validating claims, auditing compliance — all require reconstructing historical batches exactly.

## Repository Structure

```
training-data-execution-system/
├── README.md                  # This file
├── run_demo.py                # One-command execution
├── requirements.txt           # Dependencies
├── config.py                  # Configuration
│
├── tdes/                      # Main package
│   ├── __init__.py
│   ├── documents.py           # Document ingestion
│   ├── tokenizer.py           # Frozen tokenizer
│   ├── shards.py              # Shard creation + validation
│   ├── manifests.py           # Manifest generation
│   ├── firewall.py            # Evaluation blocker
│   ├── mixture.py             # Curriculum scheduler
│   ├── packing.py             # Sequence packing
│   ├── opus.py                # Data selector
│   ├── training.py            # Trainer
│   ├── ledgers.py             # Consumption + learning
│   ├── checkpoint.py          # Save/load
│   ├── replay.py              # Replay
│   ├── audit.py               # Audit
│   └── evidence.py            # Evidence generation
│
├── tests/                     # Tests
│   ├── test_shards.py
│   ├── test_manifests.py
│   ├── test_packing.py
│   ├── test_mixture.py
│   ├── test_checkpoints.py
│   ├── test_replay.py
│   └── test_evidence.py
│
├── data/                      # Raw corpus
│   ├── wikipedia/
│   ├── books/
│   ├── code/
│   └── math/
│
└── submission_artifacts/      # Generated
    ├── run.log
    ├── evidence.json
    ├── evidence.md
    ├── performance.json
    ├── manifests/
    ├── ledgers/
    ├── checkpoints/
    ├── replay/
    └── reports/
```

## Testing

```bash
pytest tests/ -v
```

Key tests:
- Tokenizer hash immutability
- Manifest validation
- Shard hash verification
- Evaluation firewall
- Packing correctness (masks, position IDs)
- Mixture compliance
- Protected floors
- OPUS decisions
- Checkpoint integrity
- Resume correctness
- Replay determinism
- Evidence generation

## Performance Targets

| Metric | Target |
|--------|--------|
| Packing utilization | >90% |
| Padding percentage | <10% |
| Tokens/sec | >1000 |
| Resume accuracy | 100% (no skip/repeat) |
| Replay hash match | 100% |

## Evidence Bundle

The system automatically generates:

**evidence.json** (machine-readable):
```json
{
  "tokenizer_integrity": {
    "status": "PASS",
    "artifact": "manifests/tokenizer_hash.txt"
  },
  "evaluation_firewall": {
    "status": "PASS",
    "artifact": "manifests/blocked_shards.json"
  },
  ...
}
```

**evidence.md** (human-readable):

| Requirement | Result | Evidence |
|-------------|--------|----------|
| Tokenizer integrity | PASS | Manifest record |
| Evaluation firewall | PASS | Blocked shard event |
| Packing correctness | PASS | Packed batch report |
| Mixture compliance | PASS | Planned vs actual |
| OPUS audit trail | PASS | Decision ledger |
| Crash recovery | PASS | Resume verification |
| Replay | PASS | Hash match |
| Learning trace | PASS | Learning ledger |
| Throughput | PASS | Performance report |

## Known Limitations

- Small corpus (demo scale)
- Tiny model (2 layers, 128 hidden)
- Limited training steps (few hundred)
- Simplified OPUS (rule-based, not learned)
- No distributed training
- No model serving

These are intentional — the goal is correctness, not scale.

## Related

- Blog post: [Training Data Execution System](https://curioustushar.github.io/blog/posts/training-data-execution-system/)
- Assignment spec: Session 6 - Training Data Execution System

## License

MIT

## Author

Tushar Gupta
