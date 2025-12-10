Palimpsa

<div align="center">
<!-- Replace with your actual logo filename if different -->
<img width="1024" height="718" alt="image" src="https://github.com/user-attachments/assets/7fa41f32-0976-42c9-8d32-2a602e56289f" />
</div>

<div align="center">
<h3>Novel Kernels for Linear Attention</h3>
<p>
A plugin for <b>flash-linear-attention</b> and <b>flame</b>.
</p>
</div>

🏗️ Developer Installation

This setup assumes you want to modify or debug fla and flame alongside Palimpsa. We recommend setting up a "workspace" folder where all three repositories live side-by-side.

1. Environment Setup

Start with a fresh environment to avoid CUDA conflicts.

# Create and activate environment
python -m venv palimpsa_env
source palimpsa_env/bin/activate

# Upgrade pip (Critical for building wheels)
pip install --upgrade pip


2. Install "The Hard Stuff" (CUDA Kernels)

We build causal-conv1d from source to ensure the CUDA version matches your local toolkit exactly.

git clone [https://github.com/Dao-AILab/causal-conv1d.git](https://github.com/Dao-AILab/causal-conv1d.git)
cd causal-conv1d
pip install .
cd ..


3. Install Research Stack (Editable Mode)

Clone the dependencies and install them in editable mode (-e).

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


4. Install Palimpsa

Finally, clone and install this repository.

git clone git@github.com:djo1996/Palimpsa.git
cd Palimpsa
pip install -e .


🚀 Usage: Training with Flame

You do not need to modify the flame source code. You can inject Palimpsa models dynamically via the config.

1. Edit your Flame Config

In your training config (e.g., flame/configs/palimpsa_1.3B.yaml), add the experimental block pointing to the integration script.

model:
  name: palimpsa        # Matches the name registered in Palimpsa
  flavor: 1.3B          # Your model configuration flavor
  # ... other standard model args ...

experimental:
  # ⚠️ CRITICAL: This path allows Flame to "see" your custom model.
  # Adjust the path relative to where you run the command.
  custom_model_path: "../Palimpsa/palimpsa/integration.py"


2. Run Training

Run the standard Flame launch command.

# Run from inside the 'flame' directory
cd ../flame
torchrun --nproc_per_node=8 main.py --config configs/palimpsa_1.3B.yaml


📂 Repository Structure

Palimpsa/
├── palimpsa/               # Source code
│   ├── layers/             # Custom layers
│   ├── models/             # HuggingFace-compatible model definitions
│   ├── ops/                # Triton/CUDA kernels
│   └── integration.py      # The bridge script for Flame
├── pyproject.toml          # Build configuration
└── README.md
