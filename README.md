# Convolutional Neural Network-Based Multi-Exposure Fusion for Single-Image Contrast Enhancement

## Overview
This repository contains the implementation of a single-image contrast enhancement framework that combines Retinex decomposition, convolutional neural networks (CNNs), and multi-exposure image fusion (MEF). The method is designed to improve the visual quality of images captured under extreme lighting conditions, such as under- and over-exposed scenarios.

### Key Contributions
1. **Retinex Decomposition and CNN Integration**: Enhances illumination and reflectance components for robust image contrast enhancement.
2. **Multi-Exposure Image Fusion (MEF)**: Incorporates multi-exposure information to improve robustness and generalization across diverse lighting conditions.
3. **Quantitative and Qualitative Validation**: Demonstrates superior performance using PSNR, SSIM, and LPIPS metrics, along with visual comparisons.

---

## Requirements

### Required Dependencies
1. **TensorFlow (v1 compatibility mode)**:
   ```bash
   pip install tensorflow==1.15
   ```
2. **PyTorch (for LPIPS)**:
   ```bash
   pip install torch torchvision
   ```
3. **LPIPS (Learned Perceptual Image Patch Similarity)**:
   ```bash
   pip install lpips
   ```
4. **Other Python Libraries**:
   ```bash
   pip install numpy opencv-python matplotlib scikit-image
   ```

Alternatively, install all dependencies using the provided requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```

### Dataset Preparation

#### Download the SICE Dataset
1. Download the SICE dataset from the official [SICE GitHub repository](https://github.com/csjcai/SICE).
2. Place the downloaded dataset in the `datasets/SICE/` directory.

#### Split the Dataset
The dataset should be split into the following proportions:
- **Train**: 80%
- **Validation**: 10%
- **Test**: 10%

Ensure that the split is consistent for reproducibility. The dataset organization should follow this structure:
```
datasets/
└── SICE/
    ├── Train/
    │   ├── Inputs/
    │   └── Labels/
    ├── Validation/
    │   ├── Inputs/
    │   └── Labels/
    └── Test/
        ├── Inputs/
        └── Labels/
```

### Repository Structure
```
├── datasets/
│   └── SICE/
├── scripts/
│   ├── Decom_train.py
│   ├── Enhance_train.py
│   ├── Decom_test.py
│   ├── Enhance_test.py
│   ├── Method_Metrics_test.py
├── matlab/
│   ├── Histogram_Analysis.m
│   ├── Intensity_Profile_Analysis.m
├── models/
│   ├── Decomposition/
│   ├── Enhancement/
├── results/
│   ├── train/
│   ├── test/
├── requirements.txt
├── README.md
```

## Usage

### Train the Model
1. Train the decomposition network:
   ```bash
   python scripts/Decom_train.py
   ```
2. Train the enhancement network:
   ```bash
   python scripts/Enhance_train.py
   ```

### Test the Model
1. Test the decomposition network:
   ```bash
   python scripts/Decom_test.py
   ```
2. Test the enhancement network:
   ```bash
   python scripts/Enhance_test.py
   ```

### Evaluate Metrics
Run the script to calculate performance metrics (PSNR, SSIM, LPIPS):
   ```bash
   python scripts/Method_Metrics_test.py
   ```

### Run MATLAB Analyses
1. Open `Histogram_Analysis.m` in MATLAB for histogram evaluation.
2. Open `Intensity_Profile_Analysis.m` in MATLAB for intensity profile analysis.

## Results

### Quantitative Metrics
The following table summarizes the quantitative performance of the proposed method:

| Metric       | Under-Exposure | Over-Exposure |
|--------------|----------------|---------------|
| PSNR         | 18.99          | 17.97         |
| SSIM         | 0.846          | 0.774         |
| LPIPS        | 0.163          | 0.220         |

### Visual Comparisons
Refer to the `results/enhance_sample` directory for visual comparisons of enhanced images across different exposure conditions.

## Citation
If you use this code, please cite our paper:
```bibtex
@article{your_paper_citation,
  title={Convolutional Neural Network-Based Multi-Exposure Fusion for Single-Image Contrast Enhancement},
  author={Sharef Hassen and Others},
  journal={Digital Signal Processing},
  year={2024},
}
```
