# nanoGPT 1.58b (BitNet-style) + Muon Experiment

## Goal
To convert Karpathy's `nanoGPT` into a ternary-weight training experiment and test whether Muon and a set of architectural changes improve convergence and final validation loss on Tiny Shakespeare.

## The Setup
* **Hardware:** AMD RX 6700 XT (`HSA_OVERRIDE_GFX_VERSION=10.3.0` was required on this machine).
* **Dataset:** Tiny Shakespeare (character-level, vocabulary size: 65).
* **Model:** "Baby GPT" (10.68M parameters: 6 layers, 6 heads, 384 embedding dim).
* **Quantization:** Replaced standard `nn.Linear` layers in the attention and MLP blocks with a custom `BitLinear` class. Token embeddings and the LM head remain full precision. RoPE replaced learned positional embeddings.

## Experiment 1: Pure AdamW Baseline
* **Optimizer:** Standard AdamW across all 1D and 2D parameters.
* **Duration:** 5,000 iterations.
* **Result:** Reached a final best validation loss of **2.618**.
* **Observation:** Samples were weak and the loss stayed far above the full-precision reference. In this configuration, BitLinear alone was not enough.

## Experiment 2: Muon + AdamW Hybrid
* **Modification:** Routed 2D matrices to Muon, 1D to AdamW.
* **Result:** Reached a validation loss of **3.1409** at iteration **250**.
* **Observation:** Early optimization moved quickly, but this was still well behind the AdamW baseline on absolute loss.

## Experiment 3: Muon + Squared ReLU
* **Modification:** Replaced `nn.GELU` with `SquaredReLU` (F.relu(x)**2).
* **Duration:** 500 iterations.
* **Result:** Reached a validation loss of **3.0645**.
* **Observation:** Stable training and slightly better early convergence than Experiment 2.

## Experiment 4: Muon + Squared ReLU + RMSNorm
* **Modification:** Swapped standard `LayerNorm` for `RMSNorm` and kept `SquaredReLU`.
* **Duration:** 500 iterations.
* **Result:** Reached a validation loss of **3.0410**.
* **Observation:** RMSNorm looked slightly better than the LayerNorm variant in these short runs.

## Experiment 5: Muon + Squared ReLU + RMSNorm + 2:4 Sparsity
* **Modification:** Implemented 2:4 semi-structured sparsity in `BitLinear` layers.
* **Duration:** 300 iterations (process stopped).
* **Result:** Reached a validation loss of **3.328**.
* **Observation:** Loss was worse than the dense variant in this run, so sparsity did not look promising at this scale.

## Experiment 6: Muon + Squared ReLU + RMSNorm + QK-Norm
* **Modification:** Added QK-Norm (RMSNorm on Queries and Keys) to stabilize the attention mechanism and reverted sparsity.
* **Duration:** 1,000 iterations.
* **Result:** Reached a validation loss of **1.5404** (was 1.6466 at 500 iterations).
* **Observation:** This was the first change that materially closed the gap to the full-precision baseline. QK-Norm appears to be important for this setup, though that conclusion is still based on a small number of local runs.

## Experiment 7: RoPE + QK-Norm + Muon (Best Run So Far)
* **Modification:** Replaced absolute positional embeddings with **Rotary Positional Embeddings (RoPE)**.
* **Duration:** 900 iterations.
* **Result:** Reached a validation loss of **1.5192** (was 1.5718 at 500 iterations).
* **Observation:** RoPE improved on the QK-Norm run by about **0.07** validation loss over a similar training window. That is encouraging, but it is still one experiment on Tiny Shakespeare, not a general claim about BitNet models.

## Next Steps / Future Research
1. **Scale Up:** Increase parameter count to 50M+ to break the 1.47 loss barrier.
2. **Value Embedding Mixing:** Incorporate multi-token prediction and value mixing from `modded-nanogpt`.
3. **AutoResearch:** Autonomous hyperparameter tuning.
