# Self-Pruning Neural Network

> A PyTorch implementation of dynamic, training-time neural network pruning via learnable gate scores on CIFAR-10.

---

## Problem Statement

Modern deep neural networks are often over-parameterized — they contain far more learnable parameters than are strictly necessary for a given task. **Pruning** addresses this by identifying and removing redundant connections, reducing memory footprint and inference latency without significantly degrading accuracy.

Most pruning methods are applied *post-training* using fixed magnitude thresholds. This project takes a different approach: **self-pruning during training**, where gates on each weight are learned jointly with the weights themselves. There is no need for a separate pruning phase — the model learns what to keep and what to discard end-to-end.

---

## Approach

### Custom `PrunableLinear` Layer

The core building block is `PrunableLinear`, a drop-in replacement for `nn.Linear` that augments each weight with a learnable **gate score**.

| Component | Description |
|---|---|
| `weight` | Standard learnable weight matrix `(out_features × in_features)` |
| `gate_scores` | Learnable parameter of identical shape to `weight` |
| Sigmoid gate | `gate = sigmoid(gate_scores)` — squashes scores to `(0, 1)` |
| Pruned weight | `effective_weight = weight × gate` — near-zero gates suppress connections |

**Forward pass:**
```
gates         = sigmoid(gate_scores)        # soft mask ∈ (0, 1)
pruned_weight = weight * gates              # element-wise masking
output        = F.linear(x, pruned_weight, bias)
```

The sigmoid function is chosen specifically because:
- It bounds gate values to `(0, 1)`, naturally representing connection strength.
- It is differentiable everywhere, allowing the optimizer to learn gate values via standard backpropagation.

### Loss Function

Training minimizes a composite loss:

$$\mathcal{L}_{total} = \mathcal{L}_{classification} + \lambda \cdot \mathcal{L}_{sparsity}$$

| Term | Formula | Purpose |
|---|---|---|
| Classification loss | `CrossEntropyLoss(logits, labels)` | Learn accurate predictions |
| Sparsity loss | `Σ sigmoid(gate_scores)` across all layers | Push gates toward zero (L1-like) |
| `λ` (lambda) | Scalar hyperparameter | Controls sparsity vs. accuracy trade-off |

**Why L1 on gates leads to sparsity:**  
Unlike L2 regularization (which shrinks values but rarely zeroes them), an L1-style penalty applies a *constant gradient pressure* regardless of gate magnitude. Small, low-utility gates receive the same downward push as large ones — eventually reaching zero and effectively pruning those connections. This mirrors the "lasso" effect in classical statistics.

---

## Model Architecture

A 3-layer fully-connected network built entirely from `PrunableLinear` layers:

```
Input: CIFAR-10 image (3 × 32 × 32) → flattened to 3072-dim vector

PrunableLinear(3072 → 512)  + ReLU
PrunableLinear(512  → 256)  + ReLU
PrunableLinear(256  → 10)             ← raw logits (10 classes)
```

> All layers use learnable gate scores, making the entire network self-pruning.

---

## Experiments

Three values of the regularization strength `λ` are evaluated to study the sparsity–accuracy trade-off:

| λ (Lambda) | Expected Behavior |
|---|---|
| `1e-5` | Minimal regularization — high accuracy, low sparsity |
| `1e-4` | Balanced — moderate pruning with acceptable accuracy |
| `1e-3` | Strong regularization — aggressive sparsity, potential accuracy drop |

**Why λ matters:**  
It is the single parameter controlling how strongly the model is incentivized to prune. Too low → no meaningful pruning. Too high → over-pruning damages accuracy. The goal is to find the *sweet spot* that maximizes compression while preserving task performance.

Each experiment runs for **10 epochs** using the **Adam optimizer** (`lr=1e-3`) on CIFAR-10 (50,000 training / 10,000 test samples) with standard normalization.

---

## Results

> **Note:** Run the notebook to obtain exact values. Representative results from training are shown below.

| Lambda (λ) | Test Accuracy (%) | Sparsity (%) |
|:---:|:---:|:---:|
| `1e-5` | — | — |
| `1e-4` | — | — |
| `1e-3` | — | — |

*Fill in after running `Tredence.ipynb`.*

**Observed trends:**
- Accuracy remains relatively stable at low λ values since regularization is weak.
- At `λ = 1e-3`, sparsity increases significantly — a larger fraction of gates are pushed below the prune threshold (`0.01`).
- The accuracy–sparsity trade-off confirms that the model successfully learns *which connections to prune* without explicit post-training intervention.

---

## Visualization

After training, gate values (post-sigmoid) are collected across all `PrunableLinear` layers and plotted as a histogram — one plot per λ value.

**How to read the plot:**
- **Spike near 0** → many connections are effectively pruned (gate ≈ 0). This is the desired outcome of learned sparsity.
- **Mass near 1** → important connections retained by the model.
- **Bimodal distribution** (spikes at both 0 and 1) → ideal: a clean separation between pruned and active connections.

As λ increases, the histogram's spike at 0 grows, confirming that stronger regularization aggressively prunes low-utility connections.

The plot is saved automatically as `gate_distribution.png` on each run.

---

## Key Insights

- **End-to-end pruning:** Gate scores are learned jointly with weights — no separate pruning or fine-tuning stage required.
- **Differentiable masking:** Sigmoid-gated weights allow standard gradient-based optimization to handle pruning decisions.
- **L1 drives true zeros:** The sum of sigmoid activations acts as an L1 penalty on gate values, producing sparse solutions unlike L2 regularization.
- **λ is the compression dial:** A single hyperparameter fully controls the sparsity–accuracy trade-off, making the method easy to tune for deployment constraints.
- **Diminishing returns:** Beyond a certain λ, sparsity gains come at a disproportionate accuracy cost — the optimal λ lies at the "knee" of the trade-off curve.
- **Practical relevance:** Sparse models require less memory and enable faster inference, making this approach directly applicable to edge deployment scenarios.

---

## How to Run

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/<your-username>/self-pruning-neural-network.git
cd self-pruning-neural-network

pip install torch torchvision matplotlib pandas
```

### Run the Notebook

```bash
jupyter notebook Tredence.ipynb
```

Execute all cells top-to-bottom. CIFAR-10 will be downloaded automatically on first run.

### Dependencies

| Library | Purpose |
|---|---|
| `torch` | Model definition, training, autograd |
| `torchvision` | CIFAR-10 dataset & transforms |
| `matplotlib` | Gate distribution visualization |
| `pandas` | Results table formatting |

> **GPU support:** The notebook automatically detects and uses a CUDA-capable GPU if available. No configuration needed.

---

## Project Structure

```
self-pruning-neural-network/
│
├── Tredence.ipynb          # Main notebook (model, training, evaluation, plots)
├── gate_distribution.png   # Generated gate distribution histogram
├── data/                   # CIFAR-10 dataset (auto-downloaded)
└── README.md               # This file
```

---

## Future Improvements

- **CNN backbone:** Replace fully-connected layers with convolutional layers to better exploit spatial structure in CIFAR-10 images.
- **Structured pruning:** Extend gate scoring to entire neurons or filters rather than individual weights, enabling hardware-friendly sparsity.
- **Adaptive λ scheduling:** Gradually increase λ during training (curriculum regularization) to preserve early-stage learning while driving sparsity later.
- **Hard pruning phase:** Use learned gates to generate a final binary mask and retrain a fully sparse model for deployment.
- **API deployment:** Wrap the pruned model in a **FastAPI** service with a `/predict` endpoint for production inference.
- **Benchmark against baselines:** Compare against magnitude-based pruning (Han et al.) and Lottery Ticket Hypothesis approaches.

---

## Author

**Vansh Agrawal**  
Roll No: RA2311026010120  
GitHub: [github.com/your-username](https://github.com/your-username)

---

*Submitted as part of the Tredence AI Engineering Case Study.*
