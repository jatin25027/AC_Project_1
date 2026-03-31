"""
Shared utility functions for the ML-based cryptanalysis project.
"""
import numpy as np
import torch
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def int_to_bits(val, nbits):
    """Convert integer to numpy bit array (MSB first)."""
    return np.array([(val >> i) & 1 for i in range(nbits - 1, -1, -1)], dtype=np.float32)

def bits_to_int(bits):
    """Convert bit array back to integer."""
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val

def int_to_words(val, word_size, num_words):
    """Convert integer to word-level array."""
    mask = (1 << word_size) - 1
    words = []
    for i in range(num_words - 1, -1, -1):
        words.append((val >> (i * word_size)) & mask)
    return np.array(words, dtype=np.float32)

def hamming_weight(val, nbits):
    """Compute Hamming weight of an integer."""
    return bin(val & ((1 << nbits) - 1)).count('1')

def xor(a, b, nbits):
    """XOR two integers with bit masking."""
    return (a ^ b) & ((1 << nbits) - 1)

def rotate_left(val, r, nbits):
    """Left rotation."""
    return ((val << r) | (val >> (nbits - r))) & ((1 << nbits) - 1)

def rotate_right(val, r, nbits):
    """Right rotation."""
    return ((val >> r) | (val << (nbits - r))) & ((1 << nbits) - 1)

def save_results(results, filepath):
    """Save results dictionary to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # Convert numpy types
    def convert(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=convert)

def load_results(filepath):
    """Load results dictionary from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)

def setup_plot_style():
    """Set up consistent, publication-quality plot style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.figsize': (10, 6),
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
    })

# Cipher registry
CIPHER_REGISTRY = {}

def register_cipher(name):
    """Decorator to register a cipher class."""
    def decorator(cls):
        CIPHER_REGISTRY[name] = cls
        return cls
    return decorator

def get_cipher(name):
    """Get a cipher class by name."""
    if name not in CIPHER_REGISTRY:
        raise ValueError(f"Unknown cipher: {name}. Available: {list(CIPHER_REGISTRY.keys())}")
    return CIPHER_REGISTRY[name]

def list_ciphers():
    """List all registered ciphers."""
    return list(CIPHER_REGISTRY.keys())
