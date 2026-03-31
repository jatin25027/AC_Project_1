"""
Input representation strategies for ML-based cryptanalysis.
Supports the 10 representations required in the project.
"""
import numpy as np

def _bits(arr, block_size):
    """Convert integer array to bit array (shape: [N, block_size])."""
    if block_size > 64:
        # Slow but safe path for >64 bits
        bits = np.zeros((len(arr), block_size), dtype=np.float32)
        for i, val in enumerate(arr):
            for j in range(block_size):
                bits[i, block_size - 1 - j] = (val >> j) & 1
        return bits
    else:
        masks = 1 << np.arange(block_size - 1, -1, -1, dtype=np.uint64)
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr, dtype=np.uint64)
        else:
            arr = arr.astype(np.uint64)
        bits = (arr[:, None] & masks) > 0
        return bits.astype(np.float32)

def prepare_representation(dataset, rep_type_id, block_size=None):
    """
    Format a dataset into one of 10 representations.
    
    1: Raw Ciphertext Pairs (C, C')
    2: Ciphertext Difference (ΔC = C ⊕ C')
    3: Concatenated (C || C')
    4: Bit-Sliced (each bit as a feature channel)
    5: Word-Level
    6: Intermediate Differences (White-Box)
    7: Masked / Noisy
    8: Joint Plaintext-Ciphertext (P, C, P⊕ΔP, C')
    9: Statistical Feature Vectors (Hamming weight, etc.)
    10: Sequential / Round-Wise (for recurrent models)
    """
    C = dataset['C'].astype(np.uint64)
    C_p = dataset['C_prime'].astype(np.uint64)
    P = dataset['P'].astype(np.uint64)
    P_p = dataset['P_prime'].astype(np.uint64)
    N = len(C)
    bs = block_size or dataset.get('block_size', 64)
    
    if rep_type_id == 1:
        # 1: Raw Ciphertext Pairs as bits: [N, 2, block_size]
        bits_c = _bits(C, bs)
        bits_cp = _bits(C_p, bs)
        return np.stack([bits_c, bits_cp], axis=1)
        
    elif rep_type_id == 2:
        # 2: Ciphertext Difference ΔC = C ⊕ C'
        diff = C ^ C_p
        return _bits(diff, bs)
        
    elif rep_type_id == 3:
        # 3: Concatenated Representation (C || C'): [N, 2 * block_size]
        bits_c = _bits(C, bs)
        bits_cp = _bits(C_p, bs)
        return np.concatenate([bits_c, bits_cp], axis=1)
        
    elif rep_type_id == 4:
        # 4: Bit-Sliced Representation: [N, 2, block_size]
        # Treat ciphertext bits as separate feature channels
        bits_c = _bits(C, bs)
        bits_cp = _bits(C_p, bs)
        return np.stack([bits_c, bits_cp], axis=1)
        
    elif rep_type_id == 5:
        # 5: Word-Level (assume 16-bit words)
        words_c = np.stack([(C >> (i * 16)) & 0xFFFF for i in range(bs//16)], axis=1)
        words_cp = np.stack([(C_p >> (i * 16)) & 0xFFFF for i in range(bs//16)], axis=1)
        # Normalize arbitrarily by max word val to be ML-friendly
        return np.concatenate([words_c, words_cp], axis=1).astype(np.float32) / 65535.0
        
    elif rep_type_id == 6:
        # 6: Intermediate Differences
        if 'intermediates_P' not in dataset:
            raise ValueError("Dataset does not contain intermediates.")
        ints_p = np.array(dataset['intermediates_P'], dtype=np.uint64)
        ints_pp = np.array(dataset['intermediates_P_prime'], dtype=np.uint64)
        diff_ints = ints_p ^ ints_pp
        # Flatten differences across all rounds
        out = []
        for i in range(diff_ints.shape[1]):
            out.append(_bits(diff_ints[:, i], bs))
        return np.concatenate(out, axis=1)
        
    elif rep_type_id == 7:
        # 7: Masked / Noisy Representation (add 10% bit flip noise to diff)
        diff = _bits(C ^ C_p, bs)
        noise = (np.random.rand(*diff.shape) < 0.1).astype(np.float32)
        return np.abs(diff - noise)  # logical XOR for float 1/0
        
    elif rep_type_id == 8:
        # 8: Joint Plaintext-Ciphertext (P, C, P⊕ΔP, C')
        bp = _bits(P, bs)
        bc = _bits(C, bs)
        bpp = _bits(P_p, bs)
        bcp = _bits(C_p, bs)
        return np.concatenate([bp, bc, bpp, bcp], axis=1)
        
    elif rep_type_id == 9:
        # 9: Statistical Feature Vectors
        diff = _bits(C ^ C_p, bs)
        # Features: Hamming weight, HW(C), HW(C_p)
        hw_diff = np.sum(diff, axis=1, keepdims=True) / bs
        hw_c = np.sum(_bits(C, bs), axis=1, keepdims=True) / bs
        hw_cp = np.sum(_bits(C_p, bs), axis=1, keepdims=True) / bs
        # simple bit correlations (e.g., bit i == bit i+1)
        diff_shift = np.roll(diff, 1, axis=1)
        correl = np.sum(diff == diff_shift, axis=1, keepdims=True) / bs
        return np.concatenate([hw_diff, hw_c, hw_cp, correl], axis=1)
        
    elif rep_type_id == 10:
        # 10: Sequential / Round-Wise (for RNNs): [N, num_rounds, block_size]
        if 'intermediates_P' not in dataset:
            raise ValueError("Dataset does not contain intermediates.")
        ints_p = np.array(dataset['intermediates_P'], dtype=np.uint64)
        ints_pp = np.array(dataset['intermediates_P_prime'], dtype=np.uint64)
        diff_ints = ints_p ^ ints_pp
        # Shape: [N, rounds, block_size]
        N, R = diff_ints.shape
        out = np.zeros((N, R, bs), dtype=np.float32)
        for i in range(R):
            out[:, i, :] = _bits(diff_ints[:, i], bs)
        return out
        
    else:
        raise ValueError(f"Unknown representation ID: {rep_type_id}")
