"""
Sanity check: Can the model learn a single fixed byte sequence?

This tests if gradients flow correctly.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from synthetic_learnability import SimpleMLP, construct_kronecker_from_bytes, algebraic_decode_raw_kronecker

print("="*80)
print("SANITY CHECK: Learning a single fixed sequence")
print("="*80)

# Single input-output pair
input_embedding = np.array([[1.0, 0.0, 0.0, 0.0]])  # Fixed input
target_bytes = b"hello"  # Fixed target

print(f"\nTarget: {target_bytes.decode('utf-8')} = {list(target_bytes)}")
print(f"Input: {input_embedding[0, :4]}")

# Create model
model = SimpleMLP(input_dim=4, hidden_dim=16, max_length=32, char_dim=256)

# Train to overfit on this single example
print(f"\nTraining to overfit on single example...")

for epoch in range(1000):
    loss = model.backward_and_update(input_embedding, [target_bytes], lr=0.1)
    
    if epoch % 100 == 0:
        # Predict
        pred_seqs, pred_lens = model.predict(input_embedding)
        pred = pred_seqs[0]
        
        # Check exact match
        exact_match = (pred == target_bytes)
        byte_match = sum(1 for i in range(min(len(pred), len(target_bytes))) if pred[i:i+1] == target_bytes[i:i+1])
        
        print(f"  Epoch {epoch:4d}: loss={loss:.4f}, pred={list(pred[:8])}, exact={exact_match}, byte_match={byte_match}/{len(target_bytes)}")

# Final test
print(f"\nFinal prediction:")
pred_seqs, pred_lens = model.predict(input_embedding)
pred = pred_seqs[0]

print(f"  Target: {list(target_bytes)}")
print(f"  Predicted: {list(pred)}")
print(f"  Match: {pred == target_bytes}")

if pred == target_bytes:
    print(f"\n✅ SUCCESS: Model learned to predict fixed sequence!")
    print(f"   Gradients are flowing correctly.")
    
    # Test Kronecker decode
    kappa = construct_kronecker_from_bytes(pred)
    decoded = algebraic_decode_raw_kronecker(kappa, 256, 32, True)
    
    if decoded == pred:
        print(f"   Kronecker decode: ✓")
    else:
        print(f"   Kronecker decode: ✗ (got {decoded})")
else:
    print(f"\n❌ FAILURE: Model cannot even learn single example")
    print(f"   Problem with gradient computation or architecture")

print("="*80)
