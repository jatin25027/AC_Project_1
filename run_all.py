"""
Main orchestrator script to run all experiments for the ML Cryptanalysis project.
It loops through all 14 ciphers, representations, models, and generates final plots.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cipher_implementations.ciphers import get_all_cipher_names
from dataset_generation.generate_dataset import generate_dataset
from input_representations.data_processing import prepare_representation
from ml_models.models import MLP, CNN, SiameseNet, MINE
from distinguisher_experiments.train_evaluate import train_model, calculate_accuracy
import torch
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Run Settings
NUM_SAMPLES = 5000
EPOCHS = 10
BATCH_SIZE = 256

ALL_CIPHERS = get_all_cipher_names()
ALL_MODELS = ['MLP', 'CNN', 'SiameseNet', 'MINE']
ALL_REPS = list(range(1, 11))
REP_LABELS = {1:"Raw", 2:"Diff", 3:"Concat", 4:"Bit-Slice", 5:"Word", 
              6:"Intermed", 7:"Noisy", 8:"Joint P-C", 9:"Stats", 10:"Sequential"}

def get_user_input(prompt, default, valid_options=None, is_int=False):
    mapping = {}
    if valid_options and not is_int:
        print(f"\n{prompt}")
        for i, opt in enumerate(valid_options, 1):
            mapping[str(i)] = opt
            print(f"  {i}. {opt}")
    else:
        valid_str = f" options: {valid_options}" if valid_options else ""
        print(f"\n{prompt}{valid_str}")
        
    print(f"  (Press Enter for default: {default} | Type names or numbers, comma-separate e.g., '1,3,4')")
    sys.stdout.flush()
    val = input(">> ").strip()
    if not val:
        val = str(default)
    
    parts = [p.strip() for p in val.split(',')]
    processed_parts = []
    
    for p in parts:
        if is_int:
            if p.isdigit():
                processed_parts.append(int(p))
        else:
            if mapping and p in mapping:
                processed_parts.append(mapping[p])
            else:
                processed_parts.append(p)

    if valid_options:
        processed_parts = [p for p in processed_parts if p in valid_options]
            
    if not processed_parts:
        print(f"  [!] Invalid selection. Falling back to default: {default}")
        def_parts = [p.strip() for p in str(default).split(',')]
        if is_int:
            processed_parts = [int(p) for p in def_parts if p.isdigit()]
        else:
            processed_parts = [p for p in def_parts if p in valid_options or (mapping and p in mapping)]
            # map defaults too if they were indices
            processed_parts = [mapping[p] if (mapping and p in mapping) else p for p in processed_parts]
        
    return processed_parts

def generate_and_evaluate(cipher_name, rounds, rep_type_id, model_type):
    # Generate Dataset
    dataset = generate_dataset(cipher_name, num_samples=NUM_SAMPLES, rounds=rounds, include_intermediates=True)
    
    # Prepare Representation
    data = prepare_representation(dataset, rep_type_id, block_size=dataset['block_size'])
    labels = dataset['labels']
    
    # Split
    split_idx = int(0.8 * len(data))
    train_data, val_data = data[:split_idx], data[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]
    
    train_loader = DataLoader(TensorDataset(torch.tensor(train_data, dtype=torch.float32), torch.tensor(train_labels, dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(val_data, dtype=torch.float32), torch.tensor(val_labels, dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize Model
    input_shape = train_data.shape[1:]
    if model_type == 'MLP':
        model = MLP(input_dim=np.prod(input_shape), hidden_dim=128)
    elif model_type == 'CNN':
        if len(input_shape) == 1:
            channels, length = 1, input_shape[0]
        else:
            channels, length = input_shape[0], input_shape[1]
            if channels > length:
                channels, length = length, channels
        model = CNN(input_channels=channels, seq_len=length)
    elif model_type == 'SiameseNet':
        if len(input_shape) > 1 and train_data.shape[1] == 2:
            model = SiameseNet(branch_dim=train_data.shape[2])
        else:
            # Fallback
            model = MLP(np.prod(input_shape))
            model_type = 'MLP'
    elif model_type == 'MINE':
        if len(input_shape) > 1 and train_data.shape[1] == 2:
            model = MINE(x_dim=train_data.shape[2], y_dim=train_data.shape[2])
        else:
            # Fallback
            model = MLP(np.prod(input_shape))
            model_type = 'MLP'
            
    acc = train_model(model, train_loader, val_loader, model_type=model_type, epochs=EPOCHS)
    
    # Store val_loader locally to return it to confusion matrix if needed
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for d, l in val_loader:
            d = d.to(DEVICE)
            if model_type == 'MINE':
                preds = (model(d[:,0,:], d[:,1,:]).squeeze() > 0).float().cpu()
            else:
                preds = (torch.sigmoid(model(d)) > 0.5).float().cpu()
            y_true.extend(l.numpy())
            y_pred.extend(preds.numpy())
            
    return acc, np.array(y_true), np.array(y_pred)

def run_confusion_matrix():
    print("\n=======================================================")
    print("  EXPERIMENT 4: Final Confusion Matrix Evaluation        ")
    print("=======================================================")
    ciphers = get_user_input("Select Cipher(s) for Confusion Matrix:", "craft", ALL_CIPHERS)
    model = get_user_input("Select Model (Pick exactly 1):", "MLP", ALL_MODELS)[0]
    rep = get_user_input("Select Representation (Pick exactly 1):", "2", ALL_REPS, is_int=True)[0]
    r = get_user_input("Select Round to test (Pick exactly 1):", "3", is_int=True)[0]

    n = len(ciphers)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)

    for idx, cipher in enumerate(ciphers):
        row, col = divmod(idx, cols)
        ax = axes[row][col]
        print(f"\n[Generating Confusion Matrix -> {cipher.upper()} | {model} | {r}R | Rep{rep}]")
        acc, y_true, y_pred = generate_and_evaluate(cipher, rounds=r, rep_type_id=rep, model_type=model)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Pred Random', 'Pred Cipher'],
                    yticklabels=['True Random', 'True Cipher'], ax=ax)
        ax.set_title(f"{cipher.upper()} (R={r}, {model}, Acc={acc:.2f})")

    # Hide any unused subplots
    for idx in range(n, rows * cols):
        row, col = divmod(idx, cols)
        axes[row][col].set_visible(False)

    plt.suptitle(f"Confusion Matrices | {model} | Rep {rep} | Round {r}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/plots/confusion_matrix.png', dpi=300)
    plt.close()
    print(">> confusion_matrix.png saved!")

def plot_dataset_statistics():
    print("\n--- Generating Dataset Distribution Graph ---")
    cipher_name = "craft"
    ds = generate_dataset(cipher_name, num_samples=20000, rounds=3)
    C, C_p = ds['C'], ds['C_prime']
    delta_C = C ^ C_p
    
    def hw(val, bits=64):
        return sum((val >> i) & 1 for i in range(bits))
    
    hw_vector = np.vectorize(lambda x: hw(x, ds['block_size']))
    hw_dist = hw_vector(delta_C)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(hw_dist, bins=range(ds['block_size'] + 2), kde=True, color='purple', stat='density')
    
    from scipy.stats import binom
    x = np.arange(0, ds['block_size'] + 1)
    y = binom.pmf(x, ds['block_size'], 0.5)
    plt.plot(x, y, 'r--', label='Ideal Random Distribution', linewidth=2)
    
    plt.title(f"Dataset Distribution: Hamming Weight of $\Delta C$ for {cipher_name.upper()}")
    plt.xlabel("Hamming Weight")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/plots/dataset_distribution.png', dpi=300)
    plt.close()
    print("  Dataset distribution graph saved.")

def run_representation_analysis():
    print("\n=======================================================")
    print("  EXPERIMENT 1: Input Representation Analysis (1-10)     ")
    print("=======================================================")
    ciphers = get_user_input("Select Cipher(s):", "craft", ALL_CIPHERS)
    models = get_user_input("Select Model(s):", "MLP", ALL_MODELS)
    rounds = get_user_input("Select Round(s):", "3", is_int=True)
    
    results = []
    reps = ALL_REPS
    
    for c in ciphers:
        for m in models:
            for r in rounds:
                print(f"\n[Evaluating Configuration -> Cipher: {c.upper()}, Model: {m}, Rounds: {r}]")
                for rep in reps:
                    acc, _, _ = generate_and_evaluate(c, rounds=r, rep_type_id=rep, model_type=m)
                    print(f"  Rep {rep} ({REP_LABELS[rep]}): {acc:.4f}")
                    results.append({
                        'Representation': REP_LABELS[rep],
                        'Accuracy': acc,
                        'Configuration': f"{c.upper()} | {m} | {r}R"
                    })
                    
    df = pd.DataFrame(results)
    plt.figure(figsize=(14, 7))
    sns.barplot(data=df, x='Representation', y='Accuracy', hue='Configuration')
    plt.axhline(0.5, color='r', linestyle='--', label='Random Guessing')
    plt.title(f"Impact of Input Representations (Multi-Config)")
    plt.xticks(rotation=45)
    plt.ylabel("Validation Accuracy")
    plt.ylim(0.4, 1.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('results/plots/representation_analysis.png', dpi=300)
    plt.close()
    print(">> representation_analysis.png saved!")

def run_model_comparison():
    print("\n=======================================================")
    print("  EXPERIMENT 2: Model Architecture Comparison            ")
    print("=======================================================")
    ciphers = get_user_input("Select Cipher(s):", "ascon", ALL_CIPHERS)
    rounds = get_user_input("Select Round(s):", "3", is_int=True)
    reps = get_user_input("Select Representation(s):", "1", ALL_REPS, is_int=True)
    
    results = []
    models = ALL_MODELS
    
    for c in ciphers:
        for r in rounds:
            for rep in reps:
                print(f"\n[Evaluating Configuration -> Cipher: {c.upper()}, Rounds: {r}, Rep: {rep}]")
                for m in models:
                    acc, _, _ = generate_and_evaluate(c, rounds=r, rep_type_id=rep, model_type=m)
                    print(f"  Model {m}: {acc:.4f}")
                    results.append({
                        'Model': m,
                        'Accuracy': acc,
                        'Configuration': f"{c.upper()} | {r}R | Rep{rep}"
                    })
                    
    df = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Model', y='Accuracy', hue='Configuration')
    plt.axhline(0.5, color='r', linestyle='--', label='Random Guessing')
    plt.title(f"Model Architecture Comparison (Multi-Config)")
    plt.ylabel("Validation Accuracy")
    plt.ylim(0.4, 1.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('results/plots/model_comparison.png', dpi=300)
    plt.close()
    print(">> model_comparison.png saved!")

def run_round_analysis():
    print("\n=======================================================")
    print("  EXPERIMENT 3: Round Limits Analysis                    ")
    print("=======================================================")
    ciphers = get_user_input("Select Cipher(s) (default is all 14):", ",".join(ALL_CIPHERS), ALL_CIPHERS)
    models = get_user_input("Select Model(s):", "MLP", ALL_MODELS)
    reps = get_user_input("Select Representation(s):", "2", ALL_REPS, is_int=True)
    # Default rounds range
    rounds_str = get_user_input("Select Rounds to sweep (comma-separated list):", "3,4,5,6,7", is_int=True)
    
    results = []
    for c in ciphers:
        for m in models:
            for rep in reps:
                print(f"\n[Testing -> {c.upper()} | Model: {m} | Rep: {rep}]")
                for r in rounds_str:
                    acc, _, _ = generate_and_evaluate(c, rounds=r, rep_type_id=rep, model_type=m)
                    print(f"  Rounds {r}: {acc:.4f}")
                    results.append({
                        'Round': r,
                        'Accuracy': acc,
                        'Configuration': f"{c.upper()} | {m} | Rep{rep}"
                    })
                    
    df = pd.DataFrame(results)
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=df, x='Round', y='Accuracy', hue='Configuration', marker='o', linewidth=2)
    plt.axhline(0.5, color='r', linestyle='--', label='Random Guessing')
    plt.title(f"Distinguisher Accuracy vs Rounds (Multi-Config)")
    plt.xlabel("Number of Rounds")
    plt.ylabel("Validation Accuracy")
    plt.ylim(0.4, 1.0)
    plt.xticks(rounds_str)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('results/plots/round_analysis_comparison.png', dpi=300)
    plt.close()
    print(">> round_analysis_comparison.png saved!")

if __name__ == "__main__":
    import shutil
    import glob
    os.makedirs('results/plots', exist_ok=True)
    for plot_file in glob.glob('results/plots/*.png'):
        os.remove(plot_file)
        
    print("="*60)
    print("  MACHINE LEARNING CRYPTANALYSIS TEST SUITE")
    print("="*60)
    plot_dataset_statistics()
    
    # Run the interactive phases
    run_representation_analysis()
    run_model_comparison()
    run_round_analysis()
    run_confusion_matrix()
    
    print("\n[SUCCESS] All experiments completed. Final interactive graphs (including Confusion Matrix) saved in results/plots/.")
