# LULC Classification using Deep Learning

## Project Overview
This project compares three pretrained deep learning models for Land Use and Land Cover (LULC) classification using the EuroSAT dataset via the TorchGeo library. The focus is on computational efficiency alongside standard accuracy metrics, with applications in precision agriculture.

## Models Compared
- ResNet-50 (pretrained on Sentinel-2 via MOCO)
- ViT-S/16 (pretrained on Sentinel-2 via DINO)
- Swin-T (pretrained on Sentinel-2 via SATLAS)

## Hardware Specifications
- Device: Apple MacBook Pro (M4 chip)
- RAM: 16GB Unified Memory
- Backend: PyTorch MPS (Metal Performance Shaders)
- OS: macOS

## Dataset
- Dataset: EuroSAT100 (via TorchGeo)
- Source: Sentinel-2 satellite imagery
- Classes: 10 LULC classes
- Total Test Images: 5400 (540 per class)
- Total Train Images: 16200 (1620 per class)
- Image Size: 64x64 pixels, 13 spectral bands

## Batch Size
- Batch size: 32 for both training and testing

## Training
- Epochs: 15
- Optimizer: Adam (lr=1e-4)
- Loss: CrossEntropyLoss

## Results

| Model     | Accuracy | Precision | Recall | F1 Score | Computation Time (s) |
|-----------|----------|-----------|--------|----------|----------------------|
| ResNet-50 | 0.80     | 0.8667    | 0.80   | 0.80     | 0.18                 |
| ViT-S/16  | 0.60     | 0.7083    | 0.60   | 0.6233   | 0.25                 |
| Swin-T    | 0.90     | 0.9167    | 0.90   | 0.8967   | 0.42                 |

## File Structure
- `resnet50.py` — ResNet-50 training and evaluation script
- `vit.py` — ViT-S/16 training and evaluation script
- `swin.py` — Swin-T training and evaluation script
- `visualize.py` — Generates comparison graphs and combined CSV
- `results/` — Contains individual and combined CSV results and graphs

## How to Run
1. Install dependencies:
pip install torch torchvision torchaudio torchgeo scikit-learn pandas matplotlib numpy

2. Run each model:
python3 resnet50.py
python3 vit.py
python3 swin.py

3. Generate graphs and combined CSV:
python3 visualize.py