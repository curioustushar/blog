"""Central configuration for Training Data Execution System."""

from dataclasses import dataclass
from typing import Dict, List, Tuple
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "submission_artifacts"
MANIFESTS_DIR = ARTIFACTS_DIR / "manifests"
LEDGERS_DIR = ARTIFACTS_DIR / "ledgers"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
REPLAY_DIR = ARTIFACTS_DIR / "replay"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

# Corpus
CORPUS_SIZES = {
    "wikipedia_train": 1000,
    "wikipedia_eval": 100,
    "books_train": 500,
    "code_train": 300,
    "math_train": 200,
}

# Tokenizer
VOCAB_SIZE = 10000
MAX_SEQ_LEN = 512

# Training
BATCH_SIZE = 8
NUM_EPOCHS = 1
LEARNING_RATE = 1e-4
CHECKPOINT_EVERY_N_STEPS = 50
CRASH_AT_STEP = 150

# Model
MODEL_CONFIG = {
    "hidden_size": 128,
    "num_layers": 2,
    "num_heads": 4,
    "vocab_size": VOCAB_SIZE,
    "max_seq_len": MAX_SEQ_LEN,
}

# Mixture Scheduler
@dataclass
class MixtureStage:
    """Curriculum stage with lane weights."""
    stage_id: int
    token_range: Tuple[int, int]
    lane_weights: Dict[str, float]
    protected_floors: Dict[str, float]

CURRICULUM = [
    MixtureStage(
        stage_id=1,
        token_range=(0, 100_000),
        lane_weights={"code": 0.10, "books": 0.50, "math": 0.20, "wikipedia": 0.20},
        protected_floors={"math": 0.05},
    ),
    MixtureStage(
        stage_id=2,
        token_range=(100_000, 300_000),
        lane_weights={"code": 0.30, "books": 0.20, "math": 0.30, "wikipedia": 0.20},
        protected_floors={"math": 0.05},
    ),
    MixtureStage(
        stage_id=3,
        token_range=(300_000, 1_000_000),
        lane_weights={"code": 0.25, "books": 0.25, "math": 0.25, "wikipedia": 0.25},
        protected_floors={"math": 0.05},
    ),
]

# OPUS
OPUS_THRESHOLDS = {
    "accept": 0.7,
    "reject": 0.3,
    "defer": (0.3, 0.7),
}

# Packing
PACKING_MAX_DOCUMENTS = 4

# Evidence
REQUIRED_EVIDENCE = [
    "tokenizer_integrity",
    "evaluation_firewall",
    "packing_correctness",
    "mixture_compliance",
    "opus_audit",
    "crash_recovery",
    "replay",
    "learning_trace",
    "throughput",
]
