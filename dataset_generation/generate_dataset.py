"""
Dataset generation for ML-based cryptanalysis.
Generates labeled (cipher vs random) differential datasets for all 14 ciphers.
"""
import numpy as np
import os, sys
import random as pyrandom

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cipher_implementations.ciphers import get_cipher, get_all_cipher_names

def generate_dataset(cipher_name, num_samples=50000, rounds=None, delta_p=None, 
                     include_intermediates=False, block_size_override=None):
    """
    Generate a labeled dataset for a given cipher.
    
    Returns:
        dict with keys: 'P', 'P_prime', 'C', 'C_prime', 'labels', 'delta_p',
                        optionally 'intermediates_P', 'intermediates_P_prime'
    """
    cipher = get_cipher(cipher_name, rounds=rounds)
    bs = block_size_override or min(cipher.BLOCK_SIZE, 64)
    mask = (1 << bs) - 1
    
    if delta_p is None:
        delta_p = pyrandom.getrandbits(bs) | 1  # Ensure non-zero
    
    n_cipher = num_samples // 2
    n_random = num_samples - n_cipher
    
    P_all, P_prime_all, C_all, C_prime_all, labels = [], [], [], [], []
    intermediates_P_list, intermediates_P_prime_list = [], []
    
    # Cipher-generated pairs (label=1)
    for _ in range(n_cipher):
        p = pyrandom.getrandbits(bs)
        p_prime = (p ^ delta_p) & mask
        if include_intermediates and hasattr(cipher, 'encrypt_with_intermediates'):
            c, ints_p = cipher.encrypt_with_intermediates(p)
            c_prime, ints_pp = cipher.encrypt_with_intermediates(p_prime)
            intermediates_P_list.append(ints_p)
            intermediates_P_prime_list.append(ints_pp)
        else:
            c = cipher.encrypt(p)
            c_prime = cipher.encrypt(p_prime)
        P_all.append(p)
        P_prime_all.append(p_prime)
        C_all.append(c & mask)
        C_prime_all.append(c_prime & mask)
        labels.append(1)
    
    # Random-generated pairs (label=0)
    for _ in range(n_random):
        p = pyrandom.getrandbits(bs)
        p_prime = (p ^ delta_p) & mask
        c = pyrandom.getrandbits(bs)
        c_prime = pyrandom.getrandbits(bs)
        P_all.append(p)
        P_prime_all.append(p_prime)
        C_all.append(c & mask)
        C_prime_all.append(c_prime & mask)
        labels.append(0)
        if include_intermediates:
            intermediates_P_list.append([pyrandom.getrandbits(bs)] * (cipher.rounds + 1 if rounds is None else rounds + 1))
            intermediates_P_prime_list.append([pyrandom.getrandbits(bs)] * (cipher.rounds + 1 if rounds is None else rounds + 1))
    
    # Shuffle
    indices = list(range(num_samples))
    pyrandom.shuffle(indices)
    
    result = {
        'P': np.array([P_all[i] for i in indices], dtype=object),
        'P_prime': np.array([P_prime_all[i] for i in indices], dtype=object),
        'C': np.array([C_all[i] for i in indices], dtype=object),
        'C_prime': np.array([C_prime_all[i] for i in indices], dtype=object),
        'labels': np.array([labels[i] for i in indices], dtype=np.int64),
        'delta_p': delta_p,
        'cipher_name': cipher_name,
        'rounds': rounds or cipher.DEFAULT_ROUNDS,
        'block_size': bs,
    }
    
    if include_intermediates:
        result['intermediates_P'] = [intermediates_P_list[i] for i in indices]
        result['intermediates_P_prime'] = [intermediates_P_prime_list[i] for i in indices]
    
    return result


def save_dataset(dataset, filepath):
    """Save dataset to numpy file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savez_compressed(filepath, **{k: v for k, v in dataset.items() if isinstance(v, np.ndarray)},
                        meta=np.array([dataset['delta_p'], dataset['rounds'], dataset['block_size']]))

def load_dataset(filepath):
    """Load dataset from numpy file."""
    data = np.load(filepath, allow_pickle=True)
    return dict(data)


if __name__ == "__main__":
    print("Generating test datasets for all 14 ciphers...")
    for name in get_all_cipher_names():
        ds = generate_dataset(name, num_samples=1000, rounds=3)
        print(f"  {name}: {len(ds['labels'])} samples, {sum(ds['labels'])} cipher / {len(ds['labels']) - sum(ds['labels'])} random")
    print("Done.")
