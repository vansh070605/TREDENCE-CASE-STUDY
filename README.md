# 🧠 Self-Pruning Neural Network

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![CIFAR-10](https://img.shields.io/badge/Dataset-CIFAR--10-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

> **Dynamic, training-time neural network pruning via learnable gate scores — no post-processing required.**

---

## 📌 Problem Statement

Modern deep neural networks are over-parameterized by design — they contain far more connections than any given task requires. While this improves expressiveness during training, it creates real-world costs: **larger model files, slower inference, and higher memory usage**.

Traditional pruning pipelines operate *after training*, requiring a separate prune → fine-tune cycle. This project implements a more elegant solution:

> **Self-pruning during training** — the network learns *which connections to keep* alongside the weights themselves, in a single end-to-end pass using learnable soft gates.

No handcrafted thresholds. No retraining. Just gradient descent.

---

## ⚙️ Approach

### Custom `PrunableLinear` Layer

`PrunableLinear` is a drop-in replacement for `nn.Linear` that attaches a learnable **gate score** to every weight element.

| Component | Description |
|:---|:---|
| `weight` | Standard learnable weight matrix `(out_features × in_features)` |
| `gate_scores` | Learnable parameter — same shape as `weight` |
| Gate | `sigmoid(gate_scores)` → continuous value in `(0, 1)` |
| Effective weight | `weight × gate` — gates near zero suppress the connection |

**Forward Pass:**
```python
gates         = sigmoid(gate_scores)   # soft mask ∈ (0, 1)
pruned_weight = weight * gates         # element-wise masking
output        = F.linear(x, pruned_weight, bias)
```

**Why Sigmoid for gating?**
- Bounds gate values to `(0, 1)` — a natural soft mask
- Differentiable everywhere → gradients flow through the gate to the optimizer
- Works seamlessly with standard backpropagation; no custom gradient tricks needed

---

### Loss Function

Training minimizes a **composite objective**:

$$\mathcal{L}_{total} = \mathcal{L}_{classification} + \lambda \cdot \mathcal{L}_{sparsity}$$

| Term | Formula | Role |
|:---|:---|:---|
| Classification loss | `CrossEntropyLoss(logits, labels)` | Drives accurate predictions |
| Sparsity loss | `Σ sigmoid(gate_scores)` across all layers | Pushes gates toward zero |
| `λ` (lambda) | Scalar hyperparameter | Compression dial — controls sparsity vs. accuracy |

**Why L1 drives true sparsity:**  
Unlike L2 regularization (which shrinks but rarely zeros values), L1 applies a *constant gradient pressure* toward zero regardless of magnitude. Low-utility gates receive the same push as large ones — eventually hitting zero and permanently pruning those connections. This is the neural network analogue of the **LASSO** in classical statistics.

---

## 🏗️ Model Architecture

A 3-layer fully-connected network where **every layer is self-pruning**:

```
Input: CIFAR-10 image (3 × 32 × 32) ──► flatten ──► 3072-dim vector

┌─────────────────────────────────────────┐
│  PrunableLinear(3072 → 512)  +  ReLU   │
│  PrunableLinear( 512 → 256)  +  ReLU   │
│  PrunableLinear( 256 →  10)            │  ← raw logits (10 classes)
└─────────────────────────────────────────┘
```

All weight matrices carry their own learnable gate masks — the entire network prunes itself during standard training.

---

## 🧪 Experiments

Three values of regularization strength `λ` are swept to map the sparsity–accuracy curve:

| λ (Lambda) | Expected Behavior |
|:---:|:---|
| `1e-5` | Minimal regularization — model retains most connections |
| `1e-4` | Balanced — moderate pruning with acceptable accuracy |
| `1e-3` | Aggressive regularization — high sparsity, potential accuracy cost |

**Training setup:**
- Optimizer: **Adam** (`lr = 1e-3`)
- Epochs: **10** per experiment
- Dataset: **CIFAR-10** — 50,000 train / 10,000 test (normalized)
- Reproducibility: `torch.manual_seed(42)`
- Hardware: Auto-selects GPU (CUDA) or CPU

---

## 📊 Results

| Lambda (λ) | Test Accuracy (%) | Sparsity (%) |
|:---:|:---:|:---:|
| `1e-5` | **56.31** | 6.25 |
| `1e-4` | **56.59** | 33.55 |
| `1e-3` | **54.31** | 57.06 |

*Gate prune threshold = 0.01 · Sparsity = % of gates below threshold*

**Key observations:**
- Accuracy is **stable across λ = 1e-5 and 1e-4** — the model compresses itself with virtually no accuracy cost.
- At `λ = 1e-3`, sparsity reaches **57%** with only a **2% accuracy drop** — a strong compression result.
- The model successfully identifies and removes redundant connections during training, without any post-hoc intervention.

> 💡 **Highlight:** Achieving 57% sparsity with only a 2-point accuracy drop on CIFAR-10 demonstrates that the gate-based self-pruning mechanism effectively identifies and eliminates redundant connections end-to-end.

---

## 📈 Visualization

After each training run, gate values (post-sigmoid) across all `PrunableLinear` layers are plotted as a histogram — one panel per λ.

**How to interpret the gate distribution:**

| Pattern | Meaning |
|:---|:---|
| Spike near **0** | Large fraction of connections pruned — desired outcome |
| Mass near **1** | Important connections the model chose to retain |
| **Bimodal** (0 and 1) | Ideal — clean separation of pruned vs. active weights |

As λ increases, the spike at 0 grows, visually confirming that stronger regularization drives more aggressive pruning. The plot is saved automatically as `gate_distribution.png`.

---

## 💡 Key Insights

- **End-to-end pruning** — gate scores are learned jointly with weights; no separate prune or fine-tune phase.
- **Differentiable masking** — sigmoid gating allows standard gradient descent to handle pruning decisions transparently.
- **L1 produces true zeros** — unlike L2, the constant gradient pressure from L1 drives low-utility gates to exactly zero.
- **λ is the compression dial** — a single hyperparameter controls the full sparsity–accuracy trade-off.
- **Diminishing returns at high λ** — sparsity gains plateau while accuracy costs accelerate; the optimal λ sits at the "knee" of the curve.
- **Deployment-ready insight** — sparse models require less memory and enable faster inference, directly applicable to edge and mobile deployment.

---

## 🚀 How to Run

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/vansh070605/TREDENCE-CASE-STUDY.git
cd TREDENCE-CASE-STUDY

pip install torch torchvision matplotlib pandas
```

### Run the Notebook

```bash
jupyter notebook Tredence.ipynb
```

Execute all cells top-to-bottom. CIFAR-10 downloads automatically on first run.

### Dependencies

| Library | Purpose |
|:---|:---|
| `torch` | Model definition, training loop, autograd |
| `torchvision` | CIFAR-10 dataset & normalization transforms |
| `matplotlib` | Gate distribution histogram |
| `pandas` | Results table formatting |

> **GPU support:** The notebook auto-detects CUDA. No manual configuration required.

---

## 📁 Project Structure

```
TREDENCE-CASE-STUDY/
│
├── Tredence.ipynb          # Main notebook — model, training, eval, plots
├── data/                   # CIFAR-10 dataset (auto-downloaded, gitignored)
├── .gitignore              # Excludes large dataset files
└── README.md               # This file
```

---

## 🔭 Future Improvements

| Idea | Description |
|:---|:---|
| **CNN backbone** | Replace FC layers with conv layers to better exploit spatial structure |
| **Structured pruning** | Gate entire neurons/filters for hardware-friendly sparsity |
| **Adaptive λ scheduling** | Gradually increase λ (curriculum regularization) to protect early training |
| **Hard pruning phase** | Convert learned soft gates into final binary masks for deployment |
| **FastAPI deployment** | Wrap pruned model in a REST API with a `/predict` endpoint |
| **Baseline comparison** | Benchmark against magnitude pruning (Han et al.) and Lottery Ticket Hypothesis |

---

## 👤 Author

**Vansh Agrawal**  
Roll No: `RA2311026010120`  
GitHub: [github.com/vansh070605](https://github.com/vansh070605)

---

*Submitted as part of the Tredence AI Engineering Internship Case Study.*
