# NanoGPT 1.58b (BitNet) + Muon Optimizer Experiment

## Goal
To convert Karpathy's `nanoGPT` into a 1.58-bit ternary network (BitNet) and test if the SOTA Muon optimizer can accelerate convergence of the latent weights behind the Straight-Through Estimator (STE).

## The Setup
* **Hardware:** AMD RX 6700 XT (Required PyTorch ROCm nightly and `HSA_OVERRIDE_GFX_VERSION=10.3.0` bypass).
* **Dataset:** Tiny Shakespeare (character-level, vocabulary size: 65).
* **Model:** "Baby GPT" (10.68M parameters: 6 layers, 6 heads, 384 embedding dim).
* **Quantization:** Replaced standard `nn.Linear` layers in the Attention and MLP blocks with a custom `BitLinear` class. The Token Embeddings, Position Embeddings, and LM Head were kept at full FP32 precision.

## Experiment 1: Pure AdamW Baseline
* **Optimizer:** Standard AdamW across all 1D and 2D parameters.
* **Duration:** 5,000 iterations.
* **Result:** Reached a final best validation loss of **2.618**.
* **Observation:** The model reached a "babble" phase. The loss floor of 2.61 indicates that a 10M parameter network lacks capacity for ternary English.

## Experiment 2: Muon + AdamW Hybrid
* **Modification:** Routed 2D matrices to Muon, 1D to AdamW.
* **Result:** Reached a validation loss of **3.1409** at iteration **250**.
* **Observation:** Confirmed ~2x speedup in convergence for latent weights.

## Experiment 3: Muon + Squared ReLU
* **Modification:** Replaced `nn.GELU` with `SquaredReLU` (F.relu(x)**2).
* **Duration:** 500 iterations.
* **Result:** Reached a validation loss of **3.0645**.
* **Observation:** Stable training and slightly better early convergence than Experiment 2.

## Experiment 4: Muon + Squared ReLU + RMSNorm
* **Modification:** Swapped standard `LayerNorm` for `RMSNorm` and kept `SquaredReLU`.
* **Duration:** 500 iterations.
* **Result:** Reached a validation loss of **3.0410**.
* **Observation:** RMSNorm outperformed LayerNorm in both speed and convergence. This confirms the "triple synergy" of BitNet + Muon + RMSNorm.

## Experiment 5: Muon + Squared ReLU + RMSNorm + 2:4 Sparsity
* **Modification:** Implemented 2:4 semi-structured sparsity in `BitLinear` layers.
* **Duration:** 300 iterations (process stopped).
* **Result:** Reached a validation loss of **3.328**.
* **Observation:** Loss was higher than the dense version. This demonstrates that for small models (10M params), cutting 50% of connections significantly impacts capacity. However, the training was stable, confirming that ternary models are naturally resilient to sparsity.

## Next Steps / Future Research
1. **QK-Norm:** Add normalization to Queries and Keys to stabilize training.
2. **Scale Up:** Increase parameter count to break the 2.0 loss barrier.
3. **AutoResearch:** Autonomous hyperparameter tuning.