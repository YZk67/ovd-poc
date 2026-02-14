# Dependencies to Run on Linux + NVIDIA GPU

All dependencies needed to run **inference** (or training) with this project on Linux and an NVIDIA GPU.

---

## 1. System / OS

- **OS**: Linux (e.g. Ubuntu 20.04/22.04)
- **NVIDIA driver**: Installed and working (`nvidia-smi` works)
- **CUDA Toolkit**: 11.7 (or 11.3 to match README PyTorch cu113; driver must support it)
  - Set `CUDA_HOME` before building, e.g. `export CUDA_HOME=/usr/local/cuda-11.7`
- **cuDNN**: Usually bundled with CUDA or installed separately; required by PyTorch for GPU

---

## 2. Python

- **Python**: 3.9 (recommended; README and env use 3.9)

---

## 3. PyTorch (with CUDA)

Install PyTorch with CUDA support. Match your CUDA version (e.g. 11.7 or 11.3).

Example for **CUDA 11.7**:

```bash
pip install torch==1.12.1+cu117 torchvision==0.13.1+cu117 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

Example for **CUDA 11.3** (as in README):

```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

Or use the [PyTorch install page](https://pytorch.org/get-started/locally/) and select Linux + CUDA version.

Verify:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.__version__)"
```

---

## 4. OpenCV

detectron2 expects OpenCV but does not install it. Install one of:

```bash
# Option A: system (if available)
sudo apt-get install python3-opencv

# Option B: pip (use the correct variant for your env)
pip install opencv-python
```

---

## 5. detectron2 (from this repo)

Build and install the bundled detectron2 (requires CUDA and `CUDA_HOME` for GPU ops):

```bash
cd /path/to/ovd-poc
export CUDA_HOME=/usr/local/cuda-11.7   # or your CUDA path
pip install -e detectron2
```

**detectron2’s Python dependencies** (installed automatically by `pip install -e detectron2`):

- Pillow >= 7.1  
- matplotlib  
- pycocotools >= 2.0.2  
- termcolor >= 1.1  
- yacs >= 0.1.8  
- tabulate  
- cloudpickle  
- tqdm > 4.29.0  
- tensorboard  
- fvcore >= 0.1.5, < 0.1.6  
- iopath >= 0.1.7, < 0.1.10  
- omegaconf >= 2.1  
- hydra-core >= 1.1  
- black  
- timm  
- packaging  

(You do not need to install these by hand if you use `pip install -e detectron2`.)

---

## 6. This project (detrex + LaMI-DETR configs)

From the repo root:

```bash
cd /path/to/ovd-poc
pip install -e .
```

This installs the **detrex** package (with CUDA extensions if `torch.cuda.is_available()` and `CUDA_HOME` are set). The root `setup.py` also lists:

- **torch**, **torchvision** (already installed above)

There is no root `requirements.txt`; other Python deps come from detectron2 and the two `pip install -e` steps.

---

## 7. Optional but useful

- **ninja**: Speeds up compilation of CUDA/C++ extensions  
  `pip install ninja` or `sudo apt-get install ninja-build`
- **scipy**: Used in some eval/analysis  
  `pip install scipy`

---

## 8. Quick install order (copy-paste)

Replace `CUDA_HOME` path and PyTorch CUDA version if needed (e.g. cu117 vs cu113).

```bash
# 1) Environment
conda create -n lami python=3.9
conda activate lami

# 2) CUDA path (adjust to your install)
export CUDA_HOME=/usr/local/cuda-11.7

# 3) PyTorch with CUDA (example: CUDA 11.7)
pip install torch==1.12.1+cu117 torchvision==0.13.1+cu117 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu117

# 4) OpenCV
pip install opencv-python

# 5) detectron2 (from repo)
cd /path/to/ovd-poc
pip install -e detectron2

# 6) This project (detrex + configs)
pip install -e .

# 7) Optional
pip install ninja scipy
```

---

## 9. Verify

```bash
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
import detectron2
import detrex
print('detectron2:', detectron2.__version__)
print('CUDA device:', torch.cuda.get_device_name(0))
print('OK')
"
```

Then run inference (example 1 GPU):

```bash
python tools/train_net.py --config-file lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py --eval-only \
  train.init_checkpoint=pretrained_models/model_0028399.pth
```

---

## Summary list (no versions)

- **System**: Linux, NVIDIA driver, CUDA Toolkit (e.g. 11.7), cuDNN  
- **Python**: 3.9  
- **PyTorch** (with CUDA) + torchvision + torchaudio  
- **OpenCV** (opencv-python or system opencv)  
- **detectron2** (from repo: `pip install -e detectron2`)  
- **This repo** (from repo: `pip install -e .`)  
- **Optional**: ninja, scipy  

All Python packages except PyTorch and OpenCV are pulled in by the two `pip install -e` steps.
