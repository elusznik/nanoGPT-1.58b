# nanoGPT-1.58b (BitNet) + Muon

![nanoGPT](assets/nanogpt.jpg)

**A modified, 1.58-bit ternary fork of Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).**

This repository replaces the standard dense floating-point math in the original nanoGPT with the state-of-the-art **BitNet b1.58** ternary architecture. We also integrated the experimental **Muon optimizer** and **QK-Norm** to overcome the gradient instability inherent in quantization-aware training. 

The original, unmodified nanoGPT documentation can be found in [original_README.md](original_README.md).

---

## 🔬 The Experiment & Findings

### The Goal
To turn Karpathy's `nanoGPT` into a 1.58-bit ternary network (BitNet) and test if the SOTA Muon optimizer can accelerate convergence of the latent weights behind the Straight-Through Estimator (STE) on a "baby" 10M parameter model.

### The Modifications (Experiment 6 - SOTA)
1.  **BitLinear Layer:** Uses the Straight-Through Estimator (STE) to force the forward pass into strictly ternary weights (`-1, 0, 1`) while keeping the backward pass gradients as full-precision floats.
2.  **Squared ReLU ($ReLU^2$):** Replaced GeLU with Squared ReLU for improved activation sparsity and faster convergence.
3.  **RMSNorm & QK-Norm:** Swapped LayerNorm for RMSNorm throughout, and added explicit normalization to Queries and Keys (QK-Norm) to stabilize the attention mechanism.
4.  **Hybrid Optimizer:** Integrated Keller Jordan's **Muon** optimizer for all 2D matrices, with **AdamW** for 1D vectors (biases, embeddings, norms).

### The Results (Tiny Shakespeare)

| Model | Optimizer | Loss (Steps) | Status |
| :--- | :--- | :--- | :--- |
| **Baseline (FP32)** | AdamW | 1.47 (5000) | Full Precision |
| **BitNet (1.58b)** | AdamW | 2.61 (5000) | Babble only |
| **BitNet (1.58b)** | **Muon + QK-Norm** | **1.54 (1000)** | **Coherent English** |

**Observation:** The SOTA configuration (Experiment 6) reached a state of coherent English dialogue **10x faster** than the AdamW baseline and is now within **0.07 of parity** with the full-precision GPT-2 benchmark.

---

## 🚀 Quick Start

### 1. Install Dependencies
```sh
pip install torch numpy transformers datasets tiktoken wandb tqdm
```
*(Note: If you are using an AMD GPU on Linux, ensure you have ROCm PyTorch installed).*

### 2. Prepare the Dataset
```sh
python data/shakespeare_char/prepare.py
```

### 3. Run the Training Loop
```sh
# Uses the hybrid Muon setup by default
python train.py config/train_shakespeare_char.py --device=cuda --compile=False
```

### 4. Sample the Model
```sh
python sample.py --out_dir=out-muon-qknorm --device=cuda --compile=False
```

## Future Work
*   Scale the network architecture to >50M parameters to cross the 1.47 loss barrier.
*   Incorporate Value Embedding Mixing and Multi-Token Prediction from [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).
*   Integrate Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) agent for autonomous tuning.