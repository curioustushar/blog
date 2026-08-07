# Training Data Execution System - Evidence Bundle

Generated: 2026-08-07T09:00:48.018099

## Summary

Total requirements: 12
Passed: 12
Failed: 0

## Evidence Table

| Requirement | Result | Evidence | Details |
|-------------|--------|----------|---------|
| tokenizer_integrity | PASS | data/tokenizer/tokenizer_hash.txt | {"hash": "931d01c581dd6c98"} |
| shard_integrity | PASS | data/shards/ | {"shard_count": 48} |
| manifest_validation | PASS | submission_artifacts/manifests/ | {"count": 48} |
| evaluation_firewall | PASS | submission_artifacts/manifests/blocked_shards.json | {"blocked": 7, "allowed": 41} |
| crash_recovery | PASS | /Users/tushargupta/projects/curioustushar.github.io/training-data-execution-system/submission_artifacts/checkpoints/ckpt_00150.json | {"expected_batch": "batch_00033", "resumed_batch": |
| replay | PASS | submission_artifacts/replay/replay_50_100.json | {"interval": "50-100", "match": true} |
| opus_audit | PASS | submission_artifacts/reports/opus_decisions.json | {"accept": 28, "reject": 60, "defer": 77, "protect |
| packing_correctness | PASS | submission_artifacts/reports/packing_report.json | {"batch_count": 33, "average_utilization": 0.8735, |
| mixture_compliance | PASS | submission_artifacts/reports/mixture.json | {"planned": {"code": 0.1, "books": 0.5, "math": 0. |
| learning_trace | PASS | submission_artifacts/ledgers/learning.jsonl | {"records": 34} |
| checkpoint_integrity | PASS | /Users/tushargupta/projects/curioustushar.github.io/training-data-execution-system/submission_artifacts/checkpoints/ckpt_00150.json | {"checkpoint_id": "ckpt_00150"} |
| throughput | PASS | submission_artifacts/performance.json | {"tokens_per_second": 8434.43, "loss_bearing_token |

## Detailed Results

### tokenizer_integrity
- **Status:** PASS
- **Artifact:** data/tokenizer/tokenizer_hash.txt
- **Timestamp:** 2026-08-07T09:00:45.398609
- **Details:** {
  "hash": "931d01c581dd6c98"
}

### shard_integrity
- **Status:** PASS
- **Artifact:** data/shards/
- **Timestamp:** 2026-08-07T09:00:45.815982
- **Details:** {
  "shard_count": 48
}

### manifest_validation
- **Status:** PASS
- **Artifact:** submission_artifacts/manifests/
- **Timestamp:** 2026-08-07T09:00:45.827690
- **Details:** {
  "count": 48
}

### evaluation_firewall
- **Status:** PASS
- **Artifact:** submission_artifacts/manifests/blocked_shards.json
- **Timestamp:** 2026-08-07T09:00:45.828427
- **Details:** {
  "blocked": 7,
  "allowed": 41
}

### crash_recovery
- **Status:** PASS
- **Artifact:** /Users/tushargupta/projects/curioustushar.github.io/training-data-execution-system/submission_artifacts/checkpoints/ckpt_00150.json
- **Timestamp:** 2026-08-07T09:00:47.588089
- **Details:** {
  "expected_batch": "batch_00033",
  "resumed_batch": "batch_00033"
}

### replay
- **Status:** PASS
- **Artifact:** submission_artifacts/replay/replay_50_100.json
- **Timestamp:** 2026-08-07T09:00:47.961188
- **Details:** {
  "interval": "50-100",
  "match": true
}

### opus_audit
- **Status:** PASS
- **Artifact:** submission_artifacts/reports/opus_decisions.json
- **Timestamp:** 2026-08-07T09:00:48.014072
- **Details:** {
  "accept": 28,
  "reject": 60,
  "defer": 77,
  "protected": 5
}

### packing_correctness
- **Status:** PASS
- **Artifact:** submission_artifacts/reports/packing_report.json
- **Timestamp:** 2026-08-07T09:00:48.014334
- **Details:** {
  "batch_count": 33,
  "average_utilization": 0.8735,
  "sample_batch": "/Users/tushargupta/projects/curioustushar.github.io/training-data-execution-system/submission_artifacts/reports/packed_batches/batch_00001.json"
}

### mixture_compliance
- **Status:** PASS
- **Artifact:** submission_artifacts/reports/mixture.json
- **Timestamp:** 2026-08-07T09:00:48.015192
- **Details:** {
  "planned": {
    "code": 0.1,
    "books": 0.5,
    "math": 0.2,
    "wikipedia": 0.2
  },
  "actual": {
    "math": 0.2727272727272727,
    "code": 0.12121212121212122,
    "books": 0.48484848484848486,
    "wikipedia": 0.12121212121212122
  },
  "compliant": true
}

### learning_trace
- **Status:** PASS
- **Artifact:** submission_artifacts/ledgers/learning.jsonl
- **Timestamp:** 2026-08-07T09:00:48.015550
- **Details:** {
  "records": 34
}

### checkpoint_integrity
- **Status:** PASS
- **Artifact:** /Users/tushargupta/projects/curioustushar.github.io/training-data-execution-system/submission_artifacts/checkpoints/ckpt_00150.json
- **Timestamp:** 2026-08-07T09:00:48.017206
- **Details:** {
  "checkpoint_id": "ckpt_00150"
}

### throughput
- **Status:** PASS
- **Artifact:** submission_artifacts/performance.json
- **Timestamp:** 2026-08-07T09:00:48.017534
- **Details:** {
  "tokens_per_second": 8434.43,
  "loss_bearing_tokens_per_second": 7367.14,
  "packing_utilization": 0.8735,
  "padding_percentage": 12.65,
  "batches_packed": 33
}
