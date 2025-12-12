## 🛠️ Installation: Core Components

We recommend setting up a unified workspace. This installation covers the core model and kernels required to run small-scale experiments (like Shakespeare).

### 1. Create Workspace & Environment
```bash
# 1. Create the working directory
mkdir Palimpsa_Lab
cd Palimpsa_Lab

# 2. Create and activate a virtual environment
# (Using Python 3.10+ is recommended)
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 2. Install Build Dependencies
To compile the custom CUDA kernels, we need PyTorch and build tools installed first.

```bash
pip install torch packaging ninja
```

### 3. Install Core Kernels
We need to compile `causal-conv1d` from source to ensure the CUDA kernels match your local setup.

```bash
git clone git@github.com:Dao-AILab/causal-conv1d.git
cd causal-conv1d
python setup.py install
cd ..
```

### 4. Install FLA & Palimpsa
```bash
# 1. Flash Linear Attention (FLA)
git clone [https://github.com/fla-org/flash-linear-attention.git](https://github.com/fla-org/flash-linear-attention.git)
cd flash-linear-attention
pip install -e .
cd ..

# 2. Palimpsa (This Repo)
git clone git@github.com:djo1996/Palimpsa.git
cd Palimpsa
pip install -e .
```
