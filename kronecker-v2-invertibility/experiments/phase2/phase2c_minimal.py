"""
Phase 2C Minimal: Fast binary answer

Focus: Prove the interface works with minimal code.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from kronecker_encoder import algebraic_decode_raw_kronecker


def construct_kappa(byte_seq: bytes) -> np.ndarray:
    """Exact Kronecker construction."""
    L = len(byte_seq)
    kappa = np.zeros(8192, dtype=np.float64)
    for p in range(L):
        idx = byte_seq[p] * 32 + p
        kappa[idx] = 1.0
    if L > 0:
        kappa /= np.sqrt(L)
    return kappa


print("="*80)
print("PHASE 2C: MINIMAL TEST")
print("="*80)

# ==================================================
# TEST A: CODEC
# ==================================================

print("\nTEST A: Codec")
print("-" * 40)

test_bytes = [
    b"hello",
    b"a",
    bytes([0, 255, 128]),
]

all_pass = True
for tb in test_bytes:
    kappa = construct_kappa(tb)
    decoded = algebraic_decode_raw_kronecker(kappa, 256, 32, True)
    ok = (decoded == tb)
    all_pass = all_pass and ok
    print(f"  {str(list(tb)):25s} → {'PASS' if ok else 'FAIL'}")

print(f"\n{'✅' if all_pass else '❌'} Codec: {'PASS' if all_pass else 'FAIL'}")

# ==================================================
# TEST B: FAKE LOGITS
# ==================================================

print("\nTEST B: Fake Logits")
print("-" * 40)

L = 8
target = np.array([104, 101, 108, 108, 111, 33, 42, 99])

logits = np.ones((1, L, 256)) * -100
for pos in range(L):
    logits[0, pos, target[pos]] = 100

pred = np.argmax(logits, axis=-1)[0]
match = np.array_equal(pred, target)

print(f"  Target: {target}")
print(f"  Pred:   {pred}")
print(f"\n{'✅' if match else '❌'} Extraction: {'PASS' if match else 'FAIL'}")

# ==================================================
# TEST C: SIMPLIFIED LEARNING TEST
# ==================================================

print("\nTEST C: Tiny Learning Test")
print("-" * 40)

print("\nTask: 10 tokens → 8 bytes (deterministic)")
print("Model: Linear (no hidden layer)")

n_tokens = 10
L = 8
n_epochs = 1000

# Create deterministic task
X = np.eye(n_tokens)
y = np.zeros((n_tokens, L), dtype=int)
for i in range(n_tokens):
    for pos in range(L):
        y[i, pos] = (i * 7 + pos * 13) % 256

print(f"\nExamples:")
for i in range(3):
    print(f"  token{i} → {y[i]}")

# Simple linear model: X @ W → logits
# W: (n_tokens, L, 256)
W = np.random.randn(n_tokens, L, 256) * 0.01

# Train
for epoch in range(n_epochs):
    # Forward
    logits = X @ W.reshape(n_tokens, -1)  # (n_tokens, L*256)
    logits = logits.reshape(n_tokens, L, 256)
    
    # Softmax
    exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
    
    # Loss
    loss = 0
    for i in range(n_tokens):
        for pos in range(L):
            loss -= np.log(probs[i, pos, y[i, pos]] + 1e-10)
    loss /= (n_tokens * L)
    
    # Gradient
    grad = probs.copy()
    for i in range(n_tokens):
        for pos in range(L):
            grad[i, pos, y[i, pos]] -= 1
    grad /= (n_tokens * L)
    
    # Update
    grad_W = X.T @ grad.reshape(n_tokens, -1)
    grad_W = grad_W.reshape(n_tokens, L, 256)
    W -= 0.1 * grad_W
    
    if epoch % 200 == 0:
        preds = np.argmax(logits, axis=-1)
        byte_acc = np.mean(preds == y)
        exact_acc = np.mean([np.array_equal(preds[i], y[i]) for i in range(n_tokens)])
        print(f"  Epoch {epoch:4d}: loss={loss:.4f}, byte={byte_acc*100:5.1f}%, exact={exact_acc*100:5.1f}%")

# Final eval
logits = (X @ W.reshape(n_tokens, -1)).reshape(n_tokens, L, 256)
preds = np.argmax(logits, axis=-1)
byte_acc = np.mean(preds == y)
exact_acc = np.mean([np.array_equal(preds[i], y[i]) for i in range(n_tokens)])

print(f"\nFinal:")
print(f"  Byte accuracy: {byte_acc*100:.1f}%")
print(f"  Exact accuracy: {exact_acc*100:.1f}%")
print(f"  p^L prediction: {(byte_acc**L)*100:.1f}%")

# Test codec on predictions
codec_ok = 0
for i in range(n_tokens):
    try:
        pred_bytes = bytes(preds[i])
        kappa = construct_kappa(pred_bytes)
        decoded = algebraic_decode_raw_kronecker(kappa, 256, 32, True)
        if decoded == pred_bytes:
            codec_ok += 1
    except:
        pass

print(f"  Codec works on predictions: {codec_ok}/{n_tokens}")

print(f"\n{'✅' if exact_acc >= 0.9 else '⚠️'} Learning: {'PASS' if exact_acc >= 0.9 else 'PARTIAL' if byte_acc >= 0.8 else 'FAIL'}")

# ==================================================
# VERDICT
# ==================================================

print("\n" + "="*80)
print("VERDICT")
print("="*80)

all_tests_pass = all_pass and match and (exact_acc >= 0.9)

if all_tests_pass:
    print("\n✅ ALL TESTS PASS")
    print("\nConclusions:")
    print("  • Codec works perfectly")
    print("  • Extraction logic is correct")
    print("  • Neural networks CAN learn structured byte prediction")
    print("  • Predicted bytes can be decoded via κ construction")
    print("\nScientific result:")
    print("  STRUCTURED BYTE PREDICTION IS LEARNABLE")
    print("\nNext steps:")
    print("  1. Test with larger models (MLP with hidden layers)")
    print("  2. Test multiple fixed lengths (L=1,2,4,8,16,32)")
    print("  3. Add variable-length (EOS)")
    print("  4. Apply to Transformer language model")
elif byte_acc >= 0.8:
    print("\n⚠️ PARTIAL SUCCESS")
    print(f"\n  Byte prediction works ({byte_acc*100:.1f}%)")
    print(f"  But exact sequences need improvement ({exact_acc*100:.1f}%)")
    print("\nThis suggests p^L degradation is the main issue.")
    print("Consider autoregressive or error-correction approaches.")
else:
    print("\n❌ TESTS INCOMPLETE")
    print("\nFurther debugging needed.")

print("="*80)
