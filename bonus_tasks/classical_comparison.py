"""
Bonus 2: Classical Comparison
Compare ML-based differential distinguishers with traditional differential analysis.
Calculates empirical probability of a given difference and compares expected data complexity.
"""
import numpy as np
import random as pyrandom
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cipher_implementations.ciphers import get_cipher

def classical_vs_ml_distinguisher(cipher_name, rounds, dp, num_samples=10000):
    print(f"Classical vs ML Comparison for {cipher_name} (R={rounds})")
    c = get_cipher(cipher_name, rounds=rounds)
    bs = min(c.BLOCK_SIZE, 64)
    mask = (1 << bs) - 1
    
    # 1. Classical: Empirical Differential Probability
    diff_counts = {}
    for _ in range(num_samples):
        # We need a random key each time for expected prob over keys, or fixed key.
        # Here we fix the key in the cipher instance so it's a fixed-key differential.
        p = pyrandom.getrandbits(bs)
        p_prime = p ^ dp
        
        c1 = c.encrypt(p)
        c2 = c.encrypt(p_prime)
        dc = c1 ^ c2
        
        diff_counts[dc] = diff_counts.get(dc, 0) + 1
        
    best_dc = max(diff_counts, key=diff_counts.get)
    best_prob = diff_counts[best_dc] / num_samples
    expected_prob = 1.0 / (1 << bs)
    
    print(f"  Classical Method:")
    print(f"    Tested {num_samples} pairs.")
    print(f"    Best Output Diff: {hex(best_dc)}")
    print(f"    Empirical Prob: {best_prob:.6f}")
    if best_prob > expected_prob:
        data_complexity = 1 / max(best_prob - expected_prob, 1e-12)
        print(f"    Estimated Data Complexity (Classical): ~{data_complexity:.0f} pairs")
    else:
        print("    No useful classic differential found (p <= expected).")
        
    # ML model's complexity is usually evaluated by the validation accuracy achieved
    # with a certain dataset size. We leave that to the main experiments folder.
    print(f"  ML Method (Reference):")
    print(f"    ML models can often find relations combining MANY differentials")
    print(f"    and typically require O(10^5) samples for 3-5 rounds.")

if __name__ == "__main__":
    classical_vs_ml_distinguisher('qarma', 2, 0x0000000000000001, 5000)
    print("Done classical comparison test.")
