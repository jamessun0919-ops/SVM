# SVM Kernel 3D Visualization Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://svm3dnchuai.streamlit.app/)

An interactive educational platform demonstrating how Support Vector Machines use Kernel Functions to transform non-linearly separable data into higher-dimensional feature spaces where linear separation becomes possible.

## Project Structure

```
SVM_Kernel_3D_Visualization_Platform/
├── manim_animations/
│   ├── phase_0_rbf_derivation.py          # RBF math foundation (6 scenes)
│   ├── phase_0_5_kernel_mapping_3d.py     # 3D kernel mapping demo (HIGHEST PRIORITY)
│   └── phase_1_svm_concepts.py            # SVM theory (6 scenes)
├── ml_implementation/
│   ├── svm_implementation.py              # Complete sklearn analysis
│   ├── report.md                          # Evaluation report
│   └── figures/                           # Generated visualizations
├── streamlit_app/
│   └── app.py                             # Interactive learning platform
├── render_all.py                          # Manim render helper
├── requirements.txt                       # Dependencies
└── README.md                              # This file
```

## Live Demo

Try it now: [https://svm3dnchuai.streamlit.app/](https://svm3dnchuai.streamlit.app/)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Streamlit App

```bash
streamlit run streamlit_app/app.py
```

### 3. Render Manim Animations

```bash
# Render all animations (medium quality)
python render_all.py

# Render Phase 0.5 only (HIGHEST PRIORITY - 3D kernel demonstration)
python render_all.py --phase 0.5

# Render in high quality for production
python render_all.py --quality high

# Render a specific scene
python render_all.py --phase 0.5 --scene Kernel_Lifting_To_3D
```

### 4. Run ML Analysis

```bash
python ml_implementation/svm_implementation.py
```

## Learning Path

| Phase | Title | Description |
|-------|-------|-------------|
| 0 | RBF Math Foundation | Distance, squared distance, Gaussian, gamma, kernel construction |
| 0.5 | **3D Kernel Mapping** | **Centerpiece** — concentric circles → 3D lift → hyperplane → margin → projection back |
| 1 | SVM Concepts | Hyperplane, margin, support vectors, soft margin, kernel SVM, RBF SVM |
| 2 | Sklearn Implementation | Real SVM training, gamma/C studies, evaluation metrics |
| 3 | Streamlit Platform | Interactive exploration with live controls |

## Key Visualizations

- **3D Scene (Phase 0.5):** True 3D Manim animation showing kernel mapping with rotating camera, hyperplane surface, support vectors, and margin visualization
- **Decision Boundaries:** 2D contour plots showing how gamma and C affect the decision surface
- **Kernel Comparison:** Side-by-side comparison of linear, polynomial, RBF, and sigmoid kernels
- **Interactive 3D Plotly:** Rotatable 3D scatter plot in the Streamlit app

## License

Educational use. Built with Manim Community Edition, scikit-learn, and Streamlit.
