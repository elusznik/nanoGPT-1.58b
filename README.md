# nanoGPT-1.58b (BitNet) + Muon

![nanoGPT](assets/nanogpt.jpg)

**A modified, 1.58-bit ternary fork of Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).**

This repository replaces the standard dense floating-point math in the original nanoGPT with the state-of-the-art **BitNet b1.58** ternary architecture. We also integrated the experimental **Muon optimizer** to overcome the gradient instability inherent in quantization-aware training. 

The original, unmodified nanoGPT documentation can be found in [original_README.md](original_README.md).

---

## 🔬 The Experiment & Findings

### The Goal
To turn Karpathy's `nanoGPT` into a 1.58-bit ternary network (BitNet) and test if the SOTA Muon optimizer can accelerate convergence of the latent weights behind the Straight-Through Estimator (STE) on a "baby" 10M parameter model.

### The Modifications
1.  **BitLinear Layer:** We injected a custom `BitLinear` class into `model.py`. This uses the Straight-Through Estimator (STE) to force the forward pass into strictly ternary weights (`-1, 0, 1`) while keeping the backward pass gradients as full-precision floats.
2.  **Architecture:** We replaced the `nn.Linear` layers in the CausalSelfAttention and MLP blocks with `BitLinear`. The Token Embeddings, Position Embeddings, and the final LM Head were left as FP32 (following standard BitNet practices).
3.  **Hybrid Optimizer:** We integrated Keller Jordan's **Muon** optimizer, assigning it to handle all 2D matrices, while falling back to standard **AdamW** for 1D vectors (biases, embeddings, LayerNorms).

### The Results (Tiny Shakespeare)

We trained the 10.68M parameter "Baby GPT" on the 1MB Tiny Shakespeare dataset for 5,000 iterations to observe the effects of the ternary math.

*   **Experiment 1: Pure AdamW Baseline:** 
    *   Trained for 5,000 iterations using standard AdamW.
    *   Bottomed out at a validation loss of **2.618**.
    *   *Observation:* The model reached a "babble" phase but never produced coherent English. A 10M parameter model severely lacks the architectural capacity to map language when restricted to ternary weights.
*   **Experiment 2: Muon + AdamW Hybrid:** 
    *   Trained using the hybrid Muon/AdamW setup.
    *   Reached a validation loss of **3.1409** at exactly iteration **250**.
    *   *Observation:* Reaching a loss of ~3.14 usually requires 600-800 iterations with a standard AdamW setup. The Muon optimizer aggressively pushed the latent weights through the quantization grid, confirming the "2x speedup" claims of the Muon architecture even in a 1.58-bit environment.

---

## 🚀 Quick Start

If you want to watch the 1.58-bit math work in real-time, you can run the tiny Shakespeare model locally.

### 1. Install Dependencies
```sh
pip install torch numpy transformers datasets tiktoken wandb tqdm
```
*(Note: If you are using an AMD GPU, you must install the ROCm version of PyTorch).*

### 2. Prepare the Dataset
Download and process the 1MB Tiny Shakespeare dataset:
```sh
python data/shakespeare_char/prepare.py
```

### 3. Run the Training Loop
Kick off the training run. By default, it will use the Muon+AdamW hybrid setup and occasionally print a small `5x5` sample of the ternary weights to your console so you can watch them update.

```sh
python train.py config/train_shakespeare_char.py --device=cuda --compile=False
```

### 4. Sample the Model
Once it generates a checkpoint in the `out-shakespeare-char` folder, you can ask the model to generate text:
```sh
python sample.py --out_dir=out-shakespeare-char --device=cuda --compile=False
```

## Future Work
*   Scale the network architecture to >50M parameters to cross the coherence threshold (loss < 2.0).
*   Replace standard `LayerNorm` with `RMSNorm` for better BitNet stability.
*   Integrate Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) agent to autonomously tune the Muon learning rate schedules.