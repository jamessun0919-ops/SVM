# SVM with RBF Kernel: Implementation Report

## Overview

This report documents the implementation and evaluation of Support Vector Machines (SVM) using the Radial Basis Function (RBF) kernel on multiple datasets. The RBF kernel enables SVM to create nonlinear decision boundaries by implicitly mapping data into a higher-dimensional feature space.

## Datasets

| Dataset | Description | Classes | Samples | Features |
|---------|-------------|---------|---------|----------|
| Circles | Concentric circles (nonlinear) | 2 | 300 | 2 |
| Moons | Interleaving half-moons | 2 | 300 | 2 |
| Classification | Complex synthetic decision boundary | 2 | 300 | 2 |
| Iris (subset) | First 2 features of Iris | 3 | 150 | 2 |

## Base Model Performance (Circles Dataset)

**Configuration:** C=1.0, gamma='scale' (auto), 70/30 train/test split

| Metric | Value |
|--------|-------|
| Accuracy | ~0.97-1.0 |
| Precision | ~0.97-1.0 |
| Recall | ~0.97-1.0 |
| F1 Score | ~0.97-1.0 |
| ROC AUC | ~0.99-1.0 |

The RBF SVM successfully separates the concentric circles, demonstrating the kernel trick in practice.

## Hyperparameter Analysis

### Gamma (γ) Parameter Study

| Gamma | Accuracy | Characteristic |
|-------|----------|----------------|
| 0.001 | Low | Underfitting - too smooth, nearly linear |
| 0.01 | Medium | Slightly better, still underfit |
| 0.1 | High | Good balance |
| 1 | High | Optimal range |
| 10 | Medium-High | Starting to overfit |
| 100 | Low | Overfitting - each point isolated |

**Gamma controls the influence radius of each training example:**
- **Small γ** → Large influence radius → Smooth, simple decision boundary → Risk of underfitting
- **Large γ** → Small influence radius → Complex, wiggly boundary → Risk of overfitting

### C Parameter Study

| C | Accuracy | Characteristic |
|---|----------|----------------|
| 0.1 | Lower | Soft margin - many support vectors |
| 1 | Good | Default balance |
| 10 | Good | Tighter margin |
| 100 | Very Good | Harder margin |
| 1000 | Similar | Near-hard margin |

**C controls the regularization / margin enforcement:**
- **Small C** → Wider margin, more tolerance for misclassification → Simpler model
- **Large C** → Narrower margin, less tolerance → More complex model

## Key Findings

1. **RBF Kernel is highly effective** for nonlinear datasets (Circles, Moons).
2. **Gamma tuning is critical** — too small underfits, too large overfits.
3. **C controls the margin** — higher C reduces tolerance for errors.
4. **Support vectors** are the critical samples that define the decision boundary.
5. **Cross-validation** confirms model stability across different data splits.

## Visualizations

- `figures/datasets_overview.png` — All datasets side by side
- `figures/circles_decision_boundary.png` — RBF SVM boundary on Circles
- `figures/gamma_study.png` — Effect of gamma on decision surface
- `figures/c_study.png` — Effect of C on decision surface
- `figures/confusion_matrix.png` — Classification confusion matrix
- `figures/roc_curve.png` — ROC curve with AUC score

## Conclusion

The RBF kernel SVM transforms non-separable data into a linearly separable form through implicit feature mapping. The kernel trick allows this transformation without explicitly computing coordinates in the high-dimensional space. Proper tuning of gamma and C is essential for optimal performance — illustrating the trade-off between bias and variance in machine learning.
