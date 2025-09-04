# Dimensionality Reduction Lab 

This repository contains a complete lab script for dimensionality reduction techniques applied to the Wine dataset and synthetic datasets. The techniques explored include:

- **PCA (Principal Component Analysis)** — from scratch & using scikit-learn  
- **LDA (Linear Discriminant Analysis)** — from scratch & using scikit-learn  
- **Kernel PCA (RBF)** — from scratch & using scikit-learn  

## Overview

This lab demonstrates key dimensionality reduction techniques in Python:

- **PCA** — unsupervised; reduces feature dimensions while preserving variance  
- **LDA** — supervised; maximizes class separability  
- **Kernel PCA** — non-linear projection for complex datasets  

It includes both **scratch implementations** and **scikit-learn implementations**, with **visualizations, decision region plots, and classification metrics**.

## How to Run

1. Make sure you have Python 3.8+ installed.  
2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
pip install -r requirements.txt
```

3. Run the lab script:

```bash
python dimensionality_reduction_lab.py
```

## Figures

Figures are saved in `./figures/` and include:

- PCA explained variance and projection plots  
- LDA projections and decision regions  
- KPCA projections for moons and circles datasets  
- Decision region plots for Logistic Regression  
## Quick Analysis (Lab Questions)

**1. Explained variance & PCA:**  
- Top eigenvalues (PCA scratch): see `dimensionality_reduction_lab.py` output  
- Components needed for ~95% variance: `n_95` (calculated in script)  

**2. PCA vs LDA:**  
- PCA: unsupervised, maximizes total variance, does not consider class labels  
- LDA: supervised, maximizes class separability  

**3. KPCA gamma effect:**  
- Small gamma → kernel nearly linear → underfitting  
- Large gamma → sensitive to local structures → may overfit  

**4. Classifier performance (example from script output):**  
```
Data shape: (..., ...) Labels: [...]
Standardized shapes: (..., ...) (..., ...)
Top eigenvalues (first 6): [...]
Cumulative variance (first 2 PCs): [...]
PCA (sklearn) - Test accuracy: ...
LDA (from scratch) - Test accuracy: ...
LDA (sklearn) - Test accuracy: ...
```

**5. Visual evidence:**  
- Check `./figures/` for scatter plots and decision regions  
- PCA separates data by variance, LDA separates classes better  
- KPCA can separate nonlinear structures (moons/circles)  
