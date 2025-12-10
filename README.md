# Palimpsa
<p align="center">
<img width="1024" height="718" alt="image" src="https://github.com/user-attachments/assets/fe74b23b-6496-4225-abc9-41d0cd17f856" />
</p>

Palimpsa provides novel kernels for Linear Attention, designed to work seamlessly with `flash-linear-attention` and `flame`.

## Installation

### 1. Prerequisite: The "Hard" Stuff (CUDA Kernels)
Because of CUDA version matching, it is safer to build `causal-conv1d` from source.

```bash
# 1. Install Causal Conv1d (Strictly Required)
git clone [https://github.com/Dao-AILab/causal-conv1d.git](https://github.com/Dao-AILab/causal-conv1d.git)
cd causal-conv1d
pip install .
cd ..

# 2. (Optional) Mamba-SSM 
# Only needed if you aren't using pure Triton kernels for Mamba2
# pip install mamba-ssm
