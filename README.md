# FaceSR-GAN — Super-Resolution of Facial Images with GANs 🧠🖼️

## Overview
**FaceSR-GAN** is a deep learning project built from scratch to upscale low-resolution facial images back into high-resolution versions. The core idea is simple: take small, blurry images of faces and teach a neural network how to "enhance" them into detailed pictures — as close as possible to the originals.

We use [CelebA](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset), a large-scale dataset of celebrity faces. Each image is processed in two versions:
- **Low-resolution input** (e.g., 16×16 or 32×32)
- **High-resolution target** (e.g., 128×128 or 256×256)


## Setup & Installation
To run the project locally, we use [`uv`](https://github.com/astral-sh/uv) for clean, reproducible Python environments:

1. **Install `uv`:**
   - On macOS / Linux:
     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```
   - On Windows:
     ```powershell
     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
     ```

2. **Install dependencies and PyTorch:**
   ```bash
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   uv sync
   ```

3. **Clone the repository:**
   ```bash
   git clone https://github.com/wiktorlewandowski9/FaceSR-GAN.git
   cd FaceSR-GAN
   ```

