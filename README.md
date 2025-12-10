<div align="center">
<img width="800" alt="Palimpsa Logo" src="https://github.com/user-attachments/assets/7fa41f32-0976-42c9-8d32-2a602e56289f" />

# Palimpsa
### Novel Kernels for Linear Attention
A research plugin for **flash-linear-attention** and **flame**.
</div>

## 🏗️ Developer Installation
This setup creates a Research Workspace where `fla`, `flame`, and `palimpsa` live side-by-side. This allows you to modify the core libraries and your custom kernels simultaneously without version conflicts.

### 1. Environment Setup
Start with a fresh environment. We explicitly pin `torchtitan` to the specific commit required by Flame.

```bash
# Create and activate environment
python -m venv palimpsa_env
source palimpsa_env/bin/activate

# Upgrade pip and install build dependencies (Critical)
pip install --upgrade pip
pip install numpy packaging ninja
```

### 2. Install Research Stack (Editable Mode)
We install the dependencies in a specific order to ensure the correct versions are used.

```bash
# 1. Install specific torchtitan commit (Required by Flame)
pip install git+[https://github.com/pytorch/torchtitan.git@0b44d4c](https://github.com/pytorch/torchtitan.git@0b44d4c)

# 2. Flash Linear Attention (The Modeling Library)
git clone [https://github.com/fla-org/flash-linear-attention.git](https://github.com/fla-org/flash-linear-attention.git)
cd flash-linear-attention
pip install -e .
cd ..

# 3. Flame (The Training Engine)
git clone [https://github.com/fla-org/flame.git](https://github.com/fla-org/flame.git)
cd flame
pip install -e .
cd ..
```

### 3. Install Palimpsa
Finally, clone and install this repository.

```bash
git clone git@github.com:djo1996/Palimpsa.git
cd Palimpsa
pip install -e .
```

---

## 🚀 Usage: Training with Flame
We provide a custom launcher (`train.py`) that automatically registers Palimpsa models into the Flame engine. **You do not need to modify the Flame source code.**

### 1. Create a Config
Create a YAML configuration file in `configs/`. You can reference standard Flame configs for hyperparameters.

**Example:** `configs/palimpsa_1.3B.yaml`
```yaml
model:
  name: palimpsa        # Matches the name registered in Palimpsa
  flavor: 1.3B          # Your model configuration flavor
  # ... other standard model args ...
  
training:
  tensor_parallel_degree: 1      # Keep model whole (TP=1)
  data_parallel_shard_degree: 8  # Split data across 8 GPUs (DP=8)
```

### 2. Run Training
**Do not** run standard `torchrun` from the flame directory. Instead, use the `train.py` provided in this repository. It loads your plugin before starting the engine.

```bash
# Run from inside the 'Palimpsa' directory
torchrun --nproc_per_node=8 train.py --config configs/palimpsa_1.3B.yaml
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
├── configs/                # Training configurations
├── train.py                # <--- The Launcher Script (Runs Flame)
├── pyproject.toml          # Build configuration
└── README.md
```

---

## 📜 Citation
If you use Palimpsa in your research, please cite:

```bibtex
@software{bonnet2025palimpsa,
  author = {Bonnet, Djohan},
  title = {Palimpsa: Novel Kernels for Linear Attention},
  year = {2025},
  url = {[https://github.com/djo1996/Palimpsa](https://github.com/djo1996/Palimpsa)}
}
```
