# Palimpsa
<p align="center">
  ![Uploading image.png…]()

</p>

Palimpsa provides novel kernels for Linear Attention, designed to work seamlessly with `flash-linear-attention` and `flame`.

## 🛠️ Installation (Developer Setup)

Follow these steps to set up a research environment where `fla`, `flame`, and `palimpsa` are all editable.

### 1. Create Environment
```bash
# Create a fresh environment
python -m venv .venv
source .venv/bin/activate

# Upgrade pip (crucial for building wheels)
pip install --upgrade pip
