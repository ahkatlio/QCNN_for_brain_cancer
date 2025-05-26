# Code Repository for Reproducing Results from: "Brain Tumor Diagnosis Using Hybrid Quantum Convolutional Neural Networks"

This repository provides the official implementation of the experiments described in the preprint:

**Brain Tumor Diagnosis Using Hybrid Quantum Convolutional Neural Networks**  
*MAZ Khan, AAO Galib, N. Innan, M.Bennai*  
Preprint available at: https://arxiv.org/abs/2401.15804


---

## Overview

This codebase has been developed to facilitate the full reproducibility of the results reported in the above manuscript. It includes all relevant components such as model training scripts, evaluation routines, configuration files, and utilities for data preprocessing and visualization.

All experiments presented in the paper were conducted using this codebase. Users are encouraged to follow the setup instructions carefully to ensure consistent and reproducible outcomes.

---

## Repository Contents

## Contents

- `Model.ipynb` – Main notebook implementing and comparing the HQCNN and classical CNN models  
-  `Resize/` – Folder for resizing MRI images for quantum-compatible input
- `Resize_Classical/` – Folder for classical CNN input format
- `brainTumorDataPublic/` – Directory containing the brain tumor dataset 
- `checkpoints/` – Saved model weights for evaluation and inference  
- `Images/` – Output plots and visualizations used in the paper  
- `Reproducibility_Report.md` – Detailed documentation of the methodology, experimental settings, and interpretation of results  
- `requirements.txt` — List of required Python packages and versions

---

## Setup Instructions

To replicate the experimental results, please proceed as follows:

### 1. Clone the Repository

```bash
git clone https://github.com/ahkatlio/QCNN_for_brain_cancer.git
cd QCNN_for_brain_cancer
```

### 2. Set up your Python environment

```bash
python -m venv venv
source venv/bin/activate  # or use `venv\Scripts\activate` on Windows
```

### 3. Install required packages

```bash
pip install -r requirements.txt
```
### 4. Run the notebooks
Execute the notebooks in the following order:

- Resize.ipynb / Resize_Classical.ipynb for preprocessing

- Model.ipynb to train and evaluate the models

---

## Citation
If you use this code, models, or findings in your own research, please cite our preprint:

```bibtex
@article{hqcnnbrain,
  title={Brain tumor diagnosis using quantum convolutional neural networks},
  author={Khan, Muhammad Al-Zafar and Galib, Abdullah Al Omar and Innan, Nouhaila and Bennai, Mohamed},
  journal={arXiv preprint arXiv:2401.15804},
  year={2025}
}
```
