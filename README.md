# Bonsai Image UI

Web UI for [Bonsai Image 4B](https://prismml.com/news/bonsai-image-4b) — a local text-to-image model by PrismML.

Auto-detects hardware and picks the right backend:

| Hardware | Backend | Speed |
|----------|---------|-------|
| NVIDIA GPU (CUDA) | gemlite ternary (1.58-bit quantized) | fast (~seconds) |
| CPU only | diffusers FP16 unpacked | slow (~30–90 min/image) |

## Requirements

- Python 3.11+
- **GPU path:** Ubuntu + NVIDIA GPU with CUDA 12.x driver
- **CPU path:** ~16 GB RAM, ~10 GB free disk (model download on first run)

### Installing CUDA (GPU path only)

Skip if you have no NVIDIA GPU.

```bash
# 1. Install NVIDIA driver
sudo apt update
sudo ubuntu-drivers install
sudo reboot

# 2. Verify driver
nvidia-smi   # note the "CUDA Version" shown top-right

# 3. Install matching CUDA toolkit (example: CUDA 12.4)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update && sudo apt install cuda-toolkit-12-4

# 4. Add to ~/.bashrc
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version   # verify
```

| Driver version | Max CUDA |
|----------------|----------|
| ≥ 550          | 12.4     |
| ≥ 525          | 12.0     |
| ≥ 470          | 11.4     |

## Setup

### 1. Clone this repo

```bash
git clone https://github.com/tillwf/bonsai
cd bonsai
```

### 2. Set up the Bonsai demo (provides model weights + Python env)

```bash
git clone https://github.com/PrismML-Eng/Bonsai-image-demo demo
cd demo
bash setup.sh                          # installs deps, downloads gemlite model (~4 GB)
cd ..
```

> **CPU only:** skip `setup.sh` if you have no GPU — the server will download the
> unpacked FP16 model (~10 GB) automatically on first start.

### 3. Start the server

```bash
./run.sh
```

Open **http://localhost:3001**

## Usage

- Enter a prompt, pick a size preset, adjust steps and seed
- **Cmd/Ctrl + Enter** to generate
- Generated images are saved to `generated/` and shown in the gallery
- Click any gallery image to view it full-size
- Use the **↓ Save** button to download, **⚄ Copy seed** to reproduce an image

## Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `PORT` | `3001` | UI server port |
| `BONSAI_VARIANT` | `auto` | GPU model: `auto` (binary if downloaded, else ternary), `binary`, `ternary` |

The server status badge in the header shows `loading model…` on startup, then `GPU` or `CPU (slow)` once ready.

## Project layout

```
app.py          # FastAPI server — model loading, /generate, /images, static UI
static/
  index.html    # Single-page UI
run.sh          # Start script (uses demo/.venv if present)
requirements.txt
generated/      # Saved PNG outputs (git-ignored)
demo/           # Bonsai demo repo (git-ignored)
```
