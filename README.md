<div align="center">

<img width="800" alt="Palimpsa Logo" src="https://github.com/user-attachments/assets/7fa41f32-0976-42c9-8d32-2a602e56289f" />

# Palimpsa

### Novel Kernels for Linear Attention
A research plugin for **[flash-linear-attention](https://github.com/fla-org/flash-linear-attention)** and **[flame](https://github.com/fla-org/flame)**.

</div>

---

## 🏗️ Developer Installation

This setup creates a **Research Workspace** where `fla`, `flame`, and `palimpsa` live side-by-side. This allows you to modify the core libraries and your custom kernels simultaneously without re-installing wheels.

### 1. Environment Setup
Start with a fresh environment to avoid CUDA version conflicts.

```bash
# Create and activate environment
python -m venv palimpsa_env
source palimpsa_env/bin/activate

# Upgrade pip (Critical for building wheels)
pip install --upgrade pip
```

### 2. Install "The Hard Stuff" (CUDA Kernels)
We build `causal-conv1d` from source to ensure the CUDA ABI matches your local toolkit exactly.

```bash
git clone [https://github.com/Dao-AILab/causal-conv1d.git](https://github.com/Dao-AILab/causal-conv1d.git)
cd causal-conv1d
pip install .
cd ..
```

### 3. Install Research Stack (Editable Mode)
Clone the dependencies and install them with `-e` (editable).

```bash
# 1. Flash Linear Attention (The Modeling Library)
git clone [https://github.com/fla-org/flash-linear-attention.git](https://github.com/fla-org/flash-linear-attention.git)
cd flash-linear-attention
pip install -e .
cd ..

# 2. Flame (The Training Engine)
git clone [https://github.com/fla-org/flame.git](https://github.com/fla-org/flame.git)
cd flame
pip install -e .
cd ..
```

### 4. Install Palimpsa
Finally, clone and install this repository.

```bash
git clone git@github.com:djo1996/Palimpsa.git
cd Palimpsa
pip install -e .
```

---

## 🚀 Usage: Training with Flame

You do not need to modify the `flame` source code. Palimpsa models are injected dynamically via the configuration file.

### 1. Edit your Flame Config
In your training config (e.g., `flame/configs/palimpsa_1.3B.yaml`), add the experimental block pointing to the integration script.

```yaml
model:
  name: palimpsa        # Matches the name registered in Palimpsa
  flavor: 1.3B          # Your model configuration flavor
  # ... other standard model args ...

experimental:
  # ⚠️ CRITICAL: This path allows Flame to "hook" your custom model.
  # Adjust the path relative to where you run the torchrun command.
  custom_model_path: "../Palimpsa/palimpsa/integration.py"
```

### 2. Run Training
Run the standard Flame launch command from inside the `flame` directory.

```bash
cd ../flame
torchrun --nproc_per_node=8 main.py --config configs/palimpsa_1.3B.yaml
```

---

## 📂 Repository Structure

```text
Palimpsa/
├── palimpsa/               # Source code
│   ├── layers/             # Custom neuromorphic/linear layers
│   ├── models/             # HuggingFace-compatible model definitions
│   ├── ops/                # Triton/CUDA kernels
│   └── integration.py      # The bridge script for Flame registry
├── pyproject.toml          # Build configuration
└── README.md
```
