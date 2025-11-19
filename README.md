# Linkterpol.ai

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)  
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-red)](https://pytorch.org/)  
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Version:** 1.2.1  

A neural network that predicts whether a photo comes from **Interpol** or **LinkedIn**, inspired by the online game [linkedin-or-interpol.com](https://linkedin-or-interpol.com).

---

## Features

- Automatic photo source prediction: Interpol vs LinkedIn.  
- GPU-accelerated training using CUDA.  
- Lightweight and easy to run locally.  

---

## Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/abgache/linkterpol.ai.git  
cd linkterpol.ai  
pip install -r requirements.txt
```

**Optional:** For GPU support, make sure you have [CUDA](https://developer.nvidia.com/cuda-downloads) installed and compatible PyTorch version:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Usage

### Training

```bash
python main.py --train --cuda
```

### Prediction

```bash
python main.py --predict --image path_to_image.jpg --cuda
```

---

## Notes

- Ensure you have a compatible GPU and CUDA installed for faster training.  
- This project is in active development; updates may come frequently.  
- Contributions are welcome! Open issues or pull requests to help improve the model.
