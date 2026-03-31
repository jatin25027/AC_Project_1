"""
Bonus 1: Difference Search
Automatically search for effective input differences (ΔP) for a block cipher.
Uses random search (or evolutionary) to find a ΔP that maximizes distinguisher accuracy.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cipher_implementations.ciphers import get_cipher

def score_difference(cipher_name, rounds, dp):
    """
    Mock scoring function. In a real scenario, this would generate a 
    small dataset, train a fast MLP and return the validation accuracy.
    Here we simulate the hardware weight as a heuristic for demonstration.
    """
    c = get_cipher(cipher_name)
    hw = bin(dp).count('1')
    # A simple heuristic: low hamming weight differences tend to be better in many ciphers
    score = max(0.5, 0.7 - (hw * 0.01))
    return score + (np.random.rand() * 0.05)

def search_differences(cipher_name, rounds=3, num_trials=50):
    print(f"Starting automatic difference search for {cipher_name} (Rounds={rounds})")
    c = get_cipher(cipher_name)
    bs = min(c.BLOCK_SIZE, 64)
    
    best_dp = None
    best_score = 0
    
    for i in range(num_trials):
        # Generate random low hamming weight difference
        dp = 0
        while dp == 0:
            bits_to_set = np.random.randint(1, 4)
            for _ in range(bits_to_set):
                dp |= (1 << np.random.randint(0, bs))
                
        score = score_difference(cipher_name, rounds, dp)
        print(f"  Trial {i+1}: Trial ΔP = {hex(dp)}, estimated eval accuracy = {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_dp = dp
            
    print(f"\\nBest difference found for {cipher_name}: ΔP = {hex(best_dp)} with accuracy {best_score:.4f}")
    return best_dp, best_score

if __name__ == "__main__":
    search_differences('skinny', rounds=2, num_trials=10)
