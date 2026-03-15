# nanoGPT-1.58b (BitNet-style) + Muon

![nanoGPT](assets/nanogpt.jpg)

**A research fork of Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) for ternary-weight training experiments.**

This repository explores a BitNet-style training setup inside nanoGPT: `BitLinear` layers with ternary forward weights via a Straight-Through Estimator (STE), plus experiments with **Muon**, **QK-Norm**, **RMSNorm**, **Squared ReLU**, and **RoPE**.

This is a small-scale research repo, not a polished or rigorously benchmarked BitNet implementation. The model still trains and runs with normal floating-point PyTorch ops; the ternary part is the forward-pass weight quantization used during training.

The original, unmodified nanoGPT documentation can be found in [original_README.md](original_README.md).

## 💻 Hardware & AMD Setup

These experiments were conducted on an **AMD Radeon RX 6700 XT** GPU. 

Training on this AMD setup required a specific environment configuration:

1.  **Python Environment:** Managed via `mise`. We used **Python 3.12**, as it is the most stable version for current ROCm PyTorch distributions.
2.  **PyTorch ROCm:** Installed the official AMD-optimized PyTorch build:
    ```sh
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2.4
    ```
3.  **The Architecture Bypass:** The RX 6700 XT (`gfx1031`) is often not explicitly supported by pre-compiled wheels. We bypassed this by forcing the HSA driver to treat the card as a `gfx1030` (the instruction-compatible 6800/6900 series model):
    ```sh
    export HSA_OVERRIDE_GFX_VERSION=10.3.0
    ```
    This environment variable is **required** for all `train.py` and `sample.py` executions on this hardware.

---

## 🔬 The Experiment & Findings

### The Goal
To turn Karpathy's `nanoGPT` into a ternary-weight training experiment and test whether Muon, QK-Norm, RMSNorm, and RoPE improve training stability and validation loss on a small Tiny Shakespeare model.

### What Changed
1.  **BitLinear:** Uses an STE so the forward pass sees ternary weights (`-1, 0, 1`) while the latent weights remain full precision.
2.  **RoPE:** Replaced learned absolute positional embeddings with rotary positional embeddings.
3.  **Squared ReLU:** Replaced GeLU with `ReLU(x)^2`.
4.  **RMSNorm and QK-Norm:** Swapped LayerNorm for RMSNorm and normalized queries and keys inside attention.
5.  **Hybrid optimizer split:** Muon for 2D parameter tensors, AdamW for 1D tensors.

### The Results (Tiny Shakespeare)

| Experiment | Optimizer | Arch Tweaks | Loss (Steps) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **FP32 Baseline** | AdamW | GPT-2 Standard | 1.47 (5000) | Reference |
| **Exp 1: BitNet Base** | AdamW | + BitLinear | 2.61 (5000) | Babble |
| **Exp 2: Muon Jump** | Muon | + Hybrid Optimizer | 3.14 (250) | Faster early descent |
| **Exp 3: ReLU² Switch** | Muon | + Squared ReLU | 3.06 (500) | Slight improvement |
| **Exp 4: RMSNorm** | Muon | + RMSNorm | 3.04 (500) | Slight improvement |
| **Exp 5: Sparsity** | Muon | + 2:4 Sparsity | 3.32 (300) | Worse in this run |
| **Exp 6: QK-Norm** | Muon | + QK-Norm | 1.64 (500) | Major improvement |
| **Exp 7: RoPE** | Muon | **+ RoPE** | **1.48 (1500)** | Best run so far |
| **Dense Control** | Muon | RoPE + RMSNorm + QK-Norm + ReLU² + `nn.Linear` | **1.48 (1000)** | Best matched control |

**Observation:** In these Tiny Shakespeare runs, QK-Norm and RoPE were the biggest improvements. Extending the ternary RoPE run out to **2000** steps brought the best validation loss down to **1.4840** at **1500** steps, after which validation flattened slightly (`1.4932` at 1750 and `1.4861` at 2000). A matched dense control using the same modernized architecture but standard `nn.Linear` reached **1.4752** at **1000** steps, so the dense model is still better, but the remaining ternary penalty is small: only about **0.021** validation loss at 1000 and **0.009** against the ternary best.

### Current Conclusion
1.  The modernized architecture matters a lot: RoPE, RMSNorm, QK-Norm, ReLU², and Muon are doing most of the heavy lifting.
2.  Standard dense `nn.Linear` still outperforms the ternary `BitLinear` version in a matched comparison.
3.  The gap is small enough that the ternary model still looks credible as a size-focused research direction, especially if it can be scaled up or exported to a real low-bit inference runtime.

---

## 🚀 Quick Start

These commands follow the main path that was actually used in this repo: prepare Tiny Shakespeare, train from scratch, then sample from a saved checkpoint.

### 1. Install Dependencies
```sh
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

### 2. Prepare the Dataset
```sh
python data/shakespeare_char/prepare.py
```

### 3. Run the Training Loop
```sh
# Required for AMD RX 6700 XT
export HSA_OVERRIDE_GFX_VERSION=10.3.0
python train.py config/train_shakespeare_char.py --device=cuda --compile=False
```

### 4. Sample the Model
```sh
python sample.py --out_dir=out-muon-rope --device=cuda --compile=False
```

## Future Work
*   Scale the network architecture to >50M parameters to cross the 1.47 loss barrier.
*   Incorporate Value Embedding Mixing and Multi-Token Prediction from [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).
*   Integrate Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) agent for autonomous tuning.
