# FaceSR-GAN — Super-Resolution of Facial Images with GANs 🧠🖼️

## Overview
**FaceSR-GAN** is a deep learning project built from scratch to upscale low-resolution facial images back into high-resolution versions. The core goal is to create a Generative Adversarial Network (GAN) that enhances facial images from 64×64 pixels to 256×256 pixels, reconstructing details and features lost in the low-resolution version.

We use [CelebA](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset), a large-scale dataset of celebrity faces. Each image is processed in two versions:
- **Low-resolution input**: 64×64 pixel facial images
- **High-resolution target**: 256×256 pixel facial images with clear details

## Network Architecture Evolution
Our GAN architecture evolved through several iterations, each with unique trade-offs:

1. **Upsample-based Generator**:
   - **Pros**: Fast training convergence
   - **Cons**: High memory consumption, detail preservation issues, training instability
   - **Results**: Quick but lower quality outputs with limited facial detail reconstruction

2. **PixelShuffle Implementation**:
   - **Pros**: More realistic image generation, significantly reduced memory footprint, flexible generator scaling
   - **Cons**: Longer training cycles, though still reasonable
   - **Results**: Improved output quality with better facial feature preservation

3. **ResidualBlock with Attention**:
   - **Pros**: Minimal memory usage, highly stable training dynamics
   - **Cons**: Slower training convergence
   - **Results**: Lower initial quality but promising architecture that would likely excel with extended training time

This evolutionary approach allowed us to understand the trade-offs between different super-resolution techniques in the context of facial image enhancement.

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
   uv sync
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Clone the repository:**
   ```bash
   git clone https://github.com/wiktorlewandowski9/FaceSR-GAN.git
   cd FaceSR-GAN
   ```

