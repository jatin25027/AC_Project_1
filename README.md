---
title: Neural Cryptanalysis Lab
emoji: 🔏
colorFrom: purple
colorTo: green
sdk: docker
pinned: true
---

# 🔏 Neural Cryptanalysis Lab

> **Interactive web application** for evaluating ML-based neural distinguishers on 14 modern lightweight block ciphers.

[![HuggingFace Space](https://img.shields.io/badge/🤗_HuggingFace-Space-yellow)](https://huggingface.co/spaces/)

---

## 🌐 Web App Features

Run all experiments **without touching the command line**:

| Experiment | Description |
|---|---|
| ① Representation Analysis | Tests all 10 input representations — which format best distinguishes cipher from random? |
| ② Model Comparison | Benchmarks MLP, CNN, SiameseNet, and MINE on the same configuration |
| ③ Round Limits Analysis | Sweeps round counts to find the security boundary of each cipher |
| ④ Confusion Matrix | Visual heatmap of true vs predicted labels for selected ciphers |
| ⑤ Dataset Distribution | Hamming weight distribution of ΔC vs ideal random |
| ⚡ Full Pipeline | Runs all 4 experiments sequentially — mirrors `python run_all.py` |

**Bonus Tasks:**
- 🔍 Differential Characteristic Search
- ⚖️ Classical Cryptanalysis vs ML Comparison
- 🔀 Transfer Learning across round counts
- 🔑 Partial Key Recovery via distinguisher scoring

### Key UI Features
- **Multi-select** ciphers, models, representations
- **Round sweep** field (comma-separated, e.g. `3,4,5,6,7`)
- **Live terminal** log during training (SSE streaming)
- **Results table** with colour-coded accuracy + CSV export
- **Pipeline mode** shows all 4 plots in a grid on completion

---

## 🛠 Running Locally

```bash
git clone https://huggingface.co/spaces/<your-username>/neural-cryptanalysis
cd neural-cryptanalysis
pip install -r requirements.txt
python3 backend_main.py
# Open http://localhost:7860
```

Or via the original CLI orchestrator:
```bash
python3 run_all.py
```

---

## 🧱 Tech Stack

- **Backend**: FastAPI + Uvicorn, background threading, SSE streaming
- **ML**: PyTorch (MLP, CNN, SiameseNet, MINE), scikit-learn
- **Frontend**: Vanilla HTML/CSS/JS — zero dependencies
- **Ciphers (14)**: SKINNY, CRAFT, ASCON, SATURNIN, GIFT-64/128, XOODOO, GIMLI, SPARKLE, KNOT, QARMA, PIPO, WARP, CHAM
- **Representations (10)**: Raw, Diff, Concat, Bit-Slice, Word, Intermed, Noisy, Joint P-C, Stats, Sequential

---

## 📄 Report

See `Neural_Cryptanalysis_Report.pdf` for the full academic write-up.
