# Robust Fair Disease Diagnosis in CT Images

Official implementation for **"Robust Fair Disease Diagnosis in CT Images"**, accepted by 3rd Workshop on New Trends in AI-Generated Media and Security (AIMS) @ CVPR 2026.

**Justin Li<sup>1*</sup>, Daniel Ding<sup>1*</sup>, Asmita Yuki Pritha<sup>2*</sup>, Aryana Hou<sup>3*</sup>, Xin Wang<sup>4</sup>, Shu Hu<sup>5†</sup>**

<sup>1</sup>Carmel High School &nbsp; <sup>2</sup>Capstone School Dhaka &nbsp; <sup>3</sup>Clarkstown High School South &nbsp; <sup>4</sup>University at Albany, SUNY &nbsp; <sup>5</sup>Purdue University

<sup>*</sup>Co-first authors &nbsp; <sup>†</sup>Corresponding author

## Overview

We propose a fairness-aware 3D classification framework that pairs a Kinetics-400 pretrained 3D ResNet-18 (adapted for single-channel CT) with a custom **FairLACVaR loss** combining Logit-Adjusted Cross-Entropy and Conditional Value-at-Risk (CVaR) fairness regularisation. An α parameter controls the accuracy–fairness trade-off, enabling practitioners to sweep from near-perfect demographic parity to maximum diagnostic performance.

## Key Results

| α | F1_male | F1_female | Score ↑ | Gap ↓ | Notes |
|---|---|---|---|---|---|
| Baseline (CE) | 0.7957 | 0.6868 | 0.7413 | 0.1089 | No fairness loss |
| LA Only | 0.8596 | 0.7375 | 0.7986 | 0.1221 | Class rebalancing only |
| CVaR Only | 0.8738 | 0.7591 | 0.8165 | 0.1148 | Fairness only |
| 0.7 | 0.7835 | 0.8394 | 0.8115 | 0.0559 | F > M |
| **0.8** | **0.8283** | **0.8522** | **0.8403** | **0.0239** | **Best overall** |
| 0.9 | 0.8910 | 0.7611 | 0.8260 | 0.1299 | Highest male F1 |

## Method

1. **Preprocessing**: CT volumes resampled to 64 slices × 256 × 256, intensity-normalised to [0, 1].
2. **Architecture**: 3D ResNet-18 with modified input convolution for grayscale input, pretrained on Kinetics-400.
3. **Loss**: FairLACVaRLoss(α, τ) — blends logit-adjusted CE (class-count-aware) with a CVaR-based fairness term that penalises worst-group risk across gender subgroups.
4. **Training**: Adam optimiser, cosine annealing scheduler, mixed-precision (GradScaler), early stopping on validation F1.
5. **Evaluation**: Gender-stratified F1, fairness gap |F1_male − F1_female|, multi-class ROC-AUC.

## Dataset

Experiments use the Fair Disease Diagnosis Database from the AIMS @ CVPR 2026 Challenge, containing 889 gender-labelled chest CT scans across 4 classes (Adenocarcinoma, Squamous Cell Carcinoma, COVID-19, Normal) with 734 training and 155 validation samples. Each scan is annotated with a binary gender attribute (Male: 404 / Female: 330 in training; Male: 77 / Female: 78 in validation).

## Requirements

```bash
pip install -r requirements.txt
```

## Quick Start

### Google Colab (Recommended)

1. Upload `Fair_Disease_Seeded.ipynb` to Google Colab
2. Select **A100 GPU** runtime
3. Place data files in `Google Drive/fair disease/`
4. Run all cells sequentially

> **Important:** Reinitialise model (Cell 4) → loss/optimiser/scheduler (Cell 5) → training loop (Cell 6) for each α run to avoid weight contamination.

### Data Preparation

```bash
python preprocess.py --source_dir /path/to/rar_files --extract_dir ./data/extracted
```

### Training

```bash
python train.py --data_dir ./data/extracted --alpha 0.7 --epochs 80
```

### Evaluation

```bash
python evaluate.py --data_dir ./data/extracted --checkpoint outputs/best_model_fair.pth
```

### Alpha Sweep

```bash
for alpha in 0.5 0.7 0.75 0.8 0.9; do
    python train.py --alpha $alpha --save_dir outputs/alpha_${alpha}
done
```

## Project Structure

```
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── preprocess.py                 # Data extraction from RAR archives
├── train.py                      # Training script with fairness loss
├── evaluate.py                   # Evaluation with gender-stratified metrics
├── dataset.py                    # LungCTDataset (3D volume loader)
├── model.py                      # 3D ResNet-18 (grayscale-adapted)
├── losses.py                     # FairLACVaRLoss (LA-CE + CVaR)
├── Fair_Disease_Seeded.ipynb     # End-to-end Colab notebook
├── configs/
│   └── default.yaml              # Default hyperparameters
├── utils/
│   ├── __init__.py
│   └── seed.py                   # Reproducibility utilities
└── fig/
    ├── fairpipeline1.png         # Pipeline overview
    └── fairresult.png            # Alpha sweep results
```

## Citation

```bibtex
@inproceedings{li2026robust,
  title={Robust Fair Disease Diagnosis in CT Images},
  author={Li, Justin and Ding, Daniel and Yuki Pritha, Asmita and Hou, Aryana and Wang, Xin and Hu, Shu},
  booktitle={Proceedings of the {IEEE/CVF} Conference on Computer Vision and Pattern Recognition ({CVPR}) Workshops},
  year={2026}
}
```

## Acknowledgements

This work is supported by the U.S. National Science Foundation (NSF) under grant IIS-2434967, and the National Artificial Intelligence Research Resource (NAIRR) Pilot and TACC Lonestar6.
