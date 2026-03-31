# Neural Cryptanalysis: Evaluating ML-based Distinguishers

This project focuses on evaluating Machine Learning-based cryptanalysis on 14 modern block ciphers proposed in the last 10 years (2016–2026). It supports 10 different input data representations and 4 separate neural architectures.

## Requirements
* Python 3.8+
* PyTorch
* NumPy
* Pandas
* Seaborn / Matplotlib
* Scipy

Install dependencies via:
```bash
pip install -r requirements.txt
```

## Running the Interactive Orchestrator
To execute the comprehensive comparisons and generate the final visualization plots, use the interactive orchestrator:

```bash
python3 run_all.py
```

The CLI will prompt you to enter indices for the following 3 Experiments:
1. **Input Representation Analysis**: Compare up to 10 representation types across chosen ciphers.
2. **Model Architecture Comparison**: Evaluate MLP, CNN, SiameseNet, or MINE.
3. **Round Limits Analysis**: Compare limits of ML distinguishers as round counts scale upwards.

*(Note: Graphs will be autosaved to `results/plots/`)*

## Running the Bonus Tasks
4 independent bonus proofs of concept are included:
1. **Heuristic Search:** `python3 bonus_tasks/difference_search.py`
2. **Classical vs ML Compare:** `python3 bonus_tasks/classical_comparison.py` 
3. **Transfer Learning:** `python3 bonus_tasks/transfer_learning.py`
4. **Key Recovery:** `python3 bonus_tasks/key_recovery.py`

## Features & Ciphers Implemented
- **Ciphers**: SKINNY, CRAFT, ASCON, SATURNIN, GIFT-64/128, XOODOO, GIMLI, SPARKE, KNOT, QARMA, PIPO, WARP, CHAM
- **Representations**: Raw Pairs, Differences, Sliced/Concat Data formats, Statistics, and Intermediates.
- **Models**: Convolutional (ResNet), Multi-Layer Perceptron, Siamese Networks, MINE.
