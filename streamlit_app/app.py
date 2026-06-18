"""
Phase 3: Streamlit Interactive Learning Platform
==================================================
Allows users to interactively explore kernel functions and SVM behavior.

Pages: Mathematical Foundation, 3D Kernel Demo, SVM Concepts, Real Model Training,
       Hyperparameter Exploration, Comparison Tool
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_circles, make_moons, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="SVM Kernel 3D Visualization Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header { font-size: 2.2rem; font-weight: 800; color: #1a1a2e;
                   text-align: center; padding: 1rem 0; }
    .sub-header { font-size: 1.4rem; font-weight: 600; color: #16213e;
                  padding: 0.5rem 0; }
    .card { background: #f8f9fa; border-radius: 10px; padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: 0.5rem 0; }
    .metric-box { background: linear-gradient(135deg, #667eea, #764ba2);
                  color: white; border-radius: 8px; padding: 1rem;
                  text-align: center; }
    .formula { font-family: 'Courier New', monospace;
               background: #2d2d2d; color: #f8f8f2; padding: 0.8rem;
               border-radius: 5px; font-size: 1.1rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">🧠 SVM Kernel 3D Visualization Platform</div>',
            unsafe_allow_html=True)
st.markdown(
    "Explore how Support Vector Machines use Kernel Functions to transform "
    "non-linearly separable data into higher-dimensional space.",
)

SIDEBAR_HEADER = """
## Controls
Adjust dataset and model parameters below.
"""


def generate_dataset(name, n_samples, noise, random_state=42):
    if name == "Circles":
        X, y = make_circles(n_samples=n_samples, noise=noise,
                            factor=0.5, random_state=random_state)
    elif name == "Moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    elif name == "Iris":
        iris = load_iris()
        X = iris.data[:, :2]
        y = iris.target
    else:
        X, y = make_circles(n_samples=n_samples, noise=noise,
                            random_state=random_state)
    return X, y


def compute_z_mapping(X):
    return X[:, 0] ** 2 + X[:, 1] ** 2


def plotly_3d_kernel_mapping(X, y, gamma_val):
    z = compute_z_mapping(X)
    fig = go.Figure()
    colors = {0: "blue", 1: "red"}
    for class_id in [0, 1]:
        mask = y == class_id
        fig.add_trace(go.Scatter3d(
            x=X[mask, 0], y=X[mask, 1], z=z[mask],
            mode="markers",
            marker=dict(size=4, color=colors[class_id],
                        opacity=0.8),
            name=f"Class {'A' if class_id == 0 else 'B'}",
        ))

    x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 20)
    y_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 20)
    xx, yy = np.meshgrid(x_range, y_range)
    zz = np.full_like(xx, 0.35)
    fig.add_trace(go.Surface(
        x=xx, y=yy, z=zz,
        colorscale=[[0, "green"], [1, "green"]],
        opacity=0.2, showscale=False, name="Hyperplane",
    ))

    fig.update_layout(
        title=f"Kernel Mapping: z = x² + y² (γ = {gamma_val})",
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
        ),
        height=600,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def plot_decision_boundary(model, X, y, scaler, title, ax):
    X_scaled = scaler.transform(X)
    x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
    y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cmap_bg = ListedColormap(["#FFAAAA", "#AAAACC", "#AAFFAA"][:len(np.unique(y))])
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_bg)
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap="bwr",
               edgecolors="k", s=30, alpha=0.8)

    if hasattr(model, "support_vectors_"):
        sv = model.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=120, facecolors="none",
                   edgecolors="yellow", linewidths=2, label="Support Vectors")
        ax.legend()

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Feature 1 (scaled)")
    ax.set_ylabel("Feature 2 (scaled)")


def sidebar_controls():
    st.sidebar.markdown(SIDEBAR_HEADER)
    dataset = st.sidebar.selectbox(
        "Dataset",
        ["Circles", "Moons", "Iris"],
        help="Circles: nonlinear concentric. Moons: interleaving crescents. Iris: classic 3-class.",
    )
    n_samples = st.sidebar.slider("Sample Size", 50, 1000, 300, 50)
    noise = st.sidebar.slider("Noise", 0.0, 0.5, 0.1, 0.05)
    gamma = st.sidebar.slider("Gamma (γ)", 0.001, 100.0, 1.0, 0.1,
                              format="%.3f")
    C = st.sidebar.slider("C (Regularization)", 0.01, 1000.0, 1.0, 0.1,
                          format="%.2f")
    kernel = st.sidebar.selectbox(
        "Kernel",
        ["rbf", "linear", "poly", "sigmoid"],
        help="RBF: radial basis (default). Linear: straight line. Poly: polynomial. Sigmoid: sigmoid.",
    )
    return dataset, n_samples, noise, gamma, C, kernel


def page_home():
    st.markdown("""
    <div class="card">
    <h3>🎯 Welcome to the SVM Kernel 3D Visualization Platform</h3>
    <p>This interactive platform helps you understand how Support Vector Machines
    use Kernel Functions to transform data into higher-dimensional spaces where
    linear separation becomes possible.</p>
    <h4>📖 Learning Path:</h4>
    <ol>
        <li><b>Mathematical Foundation</b> — Understand the RBF kernel formula</li>
        <li><b>3D Kernel Demo</b> — Watch data lift from 2D to 3D</li>
        <li><b>SVM Concepts</b> — Learn hyperplane, margin, support vectors</li>
        <li><b>Model Training</b> — Train real SVMs on your data</li>
        <li><b>Hyperparameter Studio</b> — Explore gamma, C, kernel effects</li>
        <li><b>Comparison Tool</b> — Compare kernels side by side</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-box">
        <h4>Nonlinear Data</h4>
        <p>Circles & Moons<br>need kernel trick</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-box">
        <h4>3D Feature Space</h4>
        <p>z = x² + y²<br>linear separability</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-box">
        <h4>Kernel Trick</h4>
        <p>Compute implicitly<br>O(n) not O(n²)</p>
        </div>
        """, unsafe_allow_html=True)


def page_mathematical_foundation():
    st.markdown('<div class="sub-header">📐 RBF Kernel: Mathematical Foundation</div>',
                unsafe_allow_html=True)

    gamma_val = st.slider("Interactive γ", 0.1, 5.0, 1.0, 0.1)
    dist = np.linspace(0, 5, 100)
    similarity = np.exp(-gamma_val * dist ** 2)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dist, similarity, "b-", linewidth=3, label=f"γ = {gamma_val}")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Distance ||x - x'||", fontsize=12)
    ax.set_ylabel("Similarity K(x, x')", fontsize=12)
    ax.set_title("RBF Kernel: Similarity vs Distance", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)
    plt.close()

    steps = [
        ("Euclidean Distance", r"d(x, x') = \|x - x'\|"),
        ("Squared Distance", r"\|x - x'\|^2"),
        ("Scale by Gamma", r"\gamma\|x - x'\|^2"),
        ("Exponential", r"\exp(-\gamma\|x - x'\|^2)"),
    ]
    cols = st.columns(4)
    for i, (title, formula) in enumerate(steps):
        with cols[i]:
            st.markdown(f"**{i+1}. {title}**")
            st.markdown(f"<div class='formula'>$${formula}$$</div>",
                        unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <h4>🔑 Key Intuition</h4>
    <ul>
        <li>When two points are <b>close</b> (distance ≈ 0), similarity ≈ 1</li>
        <li>When two points are <b>far</b> (distance → ∞), similarity → 0</li>
        <li><b>γ</b> controls how quickly similarity drops off with distance</li>
        <li><b>Small γ</b>: global influence, smooth decision boundary</li>
        <li><b>Large γ</b>: local influence, complex decision boundary</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


def page_3d_kernel_demo():
    st.markdown('<div class="sub-header">🌐 3D Kernel Demonstration</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Watch how the kernel mapping z = x² + y² makes linearly inseparable data "
        "separable in 3D."
    )

    dataset, n_samples, noise, gamma, C, kernel = sidebar_controls()
    X, y = generate_dataset(dataset, n_samples, noise)
    if dataset == "Iris":
        y = (y > 0).astype(int)

    fig = plotly_3d_kernel_mapping(X, y, gamma)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Original 2D Data")
        fig2d, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k", s=40)
        ax.set_title(f"{dataset} Dataset (2D)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        st.pyplot(fig2d)
        plt.close()

    with col2:
        z = compute_z_mapping(X)
        st.markdown("### 3D Feature Space")
        st.markdown(f"""
        <div class="card">
        <b>z = x² + y²</b><br>
        Inner points → low z<br>
        Outer points → high z<br>
        A plane at z ≈ constant separates them!
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        <h4>🧠 Learning Points</h4>
        <ul>
            <li>2D data is not linearly separable</li>
            <li>z = x² + y² lifts points into 3D</li>
            <li>In 3D, a flat plane separates classes</li>
            <li>The plane in 3D projects back to a circle in 2D</li>
            <li>This is the <b>kernel trick</b> in action!</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)


def page_svm_concepts():
    st.markdown('<div class="sub-header">🎯 SVM Concepts</div>',
                unsafe_allow_html=True)

    tabs = st.tabs(["Hyperplane", "Margin", "Support Vectors", "Kernel Trick"])

    with tabs[0]:
        st.markdown("""
        <div class="card">
        <h3>What is a Hyperplane?</h3>
        <p>In 2D: A line. In 3D: A plane. In n-D: An (n-1)-dimensional subspace.</p>
        <div class="formula">
        $$w^T x + b = 0$$
        </div>
        <ul>
            <li><b>w</b> = weight vector (normal to hyperplane)</li>
            <li><b>b</b> = bias term (offset from origin)</li>
            <li>Decision: sign(wᵀx + b) → +1 or -1</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        X, y = make_circles(n_samples=100, noise=0.1, factor=0.5, random_state=42)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k", s=50)
        x_vals = np.linspace(-1.5, 1.5, 100)
        for angle in [0, 30, 45, 60, 90]:
            theta = np.radians(angle)
            y_vals = np.tan(theta) * x_vals
            ax.plot(x_vals, y_vals, "--", alpha=0.3, label=f"θ={angle}°")
        ax.set_title("No Single Line Separates Circles", fontsize=14, fontweight="bold")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        st.pyplot(fig)
        plt.close()

    with tabs[1]:
        st.markdown("""
        <div class="card">
        <h3>Maximum Margin Principle</h3>
        <p>SVM doesn't just find <i>any</i> separating hyperplane — it finds the one
        with the <b>maximum margin</b>.</p>
        <div class="formula">
        $$\\max_{w,b} \\frac{2}{\\|w\\|} \\quad \\text{s.t.} \\quad y_i(w^T x_i + b) \\geq 1$$
        </div>
        <ul>
            <li>Margin = distance from hyperplane to nearest points</li>
            <li>Larger margin → better generalization</li>
            <li>C controls how strictly we enforce the margin</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with tabs[2]:
        st.markdown("""
        <div class="card">
        <h3>Support Vectors</h3>
        <p>Support Vectors are the training examples that lie on the margin boundary.
        They are the <b>only</b> points that determine the hyperplane!</p>
        <ul>
            <li>Remove non-SVs → hyperplane unchanged</li>
            <li>Remove one SV → hyperplane changes</li>
            <li>SVM is sparse — only SVs matter</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with tabs[3]:
        st.markdown("""
        <div class="card">
        <h3>The Kernel Trick</h3>
        <p>We can compute dot products in a high-dimensional feature space
        <b>without ever computing the mapping explicitly</b>.</p>
        <div class="formula">
        $$K(x, x') = \\phi(x)^T \\phi(x')$$
        </div>
        <p>The RBF kernel corresponds to an <b>infinite-dimensional</b> feature space,
        yet we compute it in O(n) time!</p>
        <h4>Available Kernels:</h4>
        <ul>
            <li><b>Linear:</b> K(x,x') = xᵀx'</li>
            <li><b>Polynomial:</b> K(x,x') = (γxᵀx' + r)^d</li>
            <li><b>RBF:</b> K(x,x') = exp(-γ||x - x'||²)</li>
            <li><b>Sigmoid:</b> K(x,x') = tanh(γxᵀx' + r)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)


@st.cache_data
def train_svm(X, y, kernel, gamma, C):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = SVC(kernel=kernel, gamma=gamma, C=C, probability=True, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler


def page_model_training():
    st.markdown('<div class="sub-header">🤖 Train Your SVM Model</div>',
                unsafe_allow_html=True)

    dataset, n_samples, noise, gamma, C, kernel = sidebar_controls()
    X, y = generate_dataset(dataset, n_samples, noise)
    if dataset == "Iris":
        y = (y > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y,
    )
    model, scaler = train_svm(X_train, y_train, kernel, gamma, C)

    y_pred = model.predict(scaler.transform(X_test))

    col1, col2 = st.columns([2, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_decision_boundary(model, X_train, y_train, scaler,
                               f"SVM ({kernel}) on {dataset}", ax)
        st.pyplot(fig)
        plt.close()

    with col2:
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        st.markdown(f"""
        <div class="metric-box">
            <h2>{acc:.3f}</h2>
            <p>Accuracy</p>
        </div>
        <br>
        <div class="metric-box">
            <h2>{f1:.3f}</h2>
            <p>F1 Score</p>
        </div>
        <br>
        <div class="card">
            <b>Model Info</b><br>
            Kernel: {kernel}<br>
            Gamma: {gamma}<br>
            C: {C}<br>
            Support Vectors: {len(model.support_vectors_)}<br>
            Training samples: {len(X_train)}<br>
            Test samples: {len(X_test)}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    im = ax_cm.imshow(cm, interpolation="nearest", cmap="Blues")
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xlabel("Predicted", fontsize=12)
    ax_cm.set_ylabel("Actual", fontsize=12)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16)
    st.pyplot(fig_cm)
    plt.close()

    with st.expander("📋 Classification Report"):
        st.text(classification_report(y_test, y_pred))


def page_hyperparameter_studio():
    st.markdown('<div class="sub-header">🔬 Hyperparameter Exploration Studio</div>',
                unsafe_allow_html=True)

    dataset, n_samples, noise, gamma, C, kernel = sidebar_controls()
    X, y = generate_dataset(dataset, n_samples, noise)
    if dataset == "Iris":
        y = (y > 0).astype(int)

    st.markdown("### Gamma Sweep (fixed C)")
    gamma_values = [0.001, 0.01, 0.1, 1, 10, 100]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, (ax, g) in enumerate(zip(axes, gamma_values)):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y,
        )
        m, s = train_svm(X_tr, y_tr, kernel, g, C)
        y_pr = m.predict(s.transform(X_te))
        acc = accuracy_score(y_te, y_pr)
        plot_decision_boundary(m, X_tr, y_tr, s, f"γ={g} (acc={acc:.3f})", ax)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("### C Sweep (fixed γ)")
    c_values = [0.01, 0.1, 1, 10, 100, 1000]
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
    axes2 = axes2.flatten()

    for i, (ax, c_val) in enumerate(zip(axes2, c_values)):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y,
        )
        m, s = train_svm(X_tr, y_tr, kernel, gamma, c_val)
        y_pr = m.predict(s.transform(X_te))
        acc = accuracy_score(y_te, y_pr)
        plot_decision_boundary(m, X_tr, y_tr, s, f"C={c_val} (acc={acc:.3f})", ax)

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.markdown("""
    <div class="card">
    <h4>📊 Key Takeaways</h4>
    <ul>
        <li><b>γ too small</b> → Underfitting: boundary too smooth, low accuracy</li>
        <li><b>γ too large</b> → Overfitting: boundary too complex, isolated regions</li>
        <li><b>C too small</b> → Underfitting: margin too wide, many errors</li>
        <li><b>C too large</b> → Overfitting: margin too tight, sensitive to noise</li>
        <li><b>Goal</b>: Find the sweet spot where validation accuracy peaks</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


def page_comparison_tool():
    st.markdown('<div class="sub-header">⚖️ Kernel Comparison Tool</div>',
                unsafe_allow_html=True)

    dataset, n_samples, noise, gamma, C, _ = sidebar_controls()
    X, y = generate_dataset(dataset, n_samples, noise)
    if dataset == "Iris":
        y = (y > 0).astype(int)

    kernels = ["linear", "poly", "rbf", "sigmoid"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    results = []
    for ax, kernel in zip(axes, kernels):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y,
        )
        m, s = train_svm(X_tr, y_tr, kernel, gamma, C)
        y_pr = m.predict(s.transform(X_te))
        acc = accuracy_score(y_te, y_pr)
        n_sv = len(m.support_vectors_) if hasattr(m, "support_vectors_") else 0
        results.append({
            "Kernel": kernel,
            "Accuracy": f"{acc:.4f}",
            "Support Vectors": n_sv,
        })
        plot_decision_boundary(m, X_tr, y_tr, s,
                               f"{kernel.upper()} (acc={acc:.4f})", ax)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("### Performance Comparison")
    st.dataframe(pd.DataFrame(results), use_container_width=True)

    st.markdown("""
    <div class="card">
    <h4>🧠 When to Use Each Kernel</h4>
    <table>
        <tr><th>Kernel</th><th>Best For</th><th>Decision Boundary</th></tr>
        <tr><td><b>Linear</b></td><td>Linearly separable data, high-dimensional sparse data (text)</td><td>Straight line/plane</td></tr>
        <tr><td><b>Polynomial</b></td><td>Data with polynomial relationships</td><td>Curved, polynomial shape</td></tr>
        <tr><td><b>RBF</b></td><td>Most nonlinear real-world data (default choice)</td><td>Any smooth shape, universal approximator</td></tr>
        <tr><td><b>Sigmoid</b></td><td>Neural network-like behavior</td><td>Similar to RBF but less stable</td></tr>
    </table>
    </div>
    """, unsafe_allow_html=True)


def page_code_viewer():
    st.markdown('<div class="sub-header">📄 Code Viewer</div>',
                unsafe_allow_html=True)
    st.markdown("View the Python implementation code for the SVM training pipeline.")
    with st.expander("View Training Code", expanded=True):
        st.code("""
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def train_rbf_svm(X, y, C=1.0, gamma='scale'):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y,
    )

    # Scale features (crucial for SVM!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train RBF SVM
    model = SVC(
        kernel='rbf',
        C=C,
        gamma=gamma,
        probability=True,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    accuracy = model.score(X_test_scaled, y_test)
    return model, scaler, accuracy
        """, language="python")


def main():
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "📌 Navigate",
        [
            "🏠 Home",
            "📐 Math Foundation",
            "🌐 3D Kernel Demo",
            "🎯 SVM Concepts",
            "🤖 Model Training",
            "🔬 Hyperparameter Studio",
            "⚖️ Kernel Comparison",
            "📄 Code Viewer",
        ],
    )

    page_map = {
        "🏠 Home": page_home,
        "📐 Math Foundation": page_mathematical_foundation,
        "🌐 3D Kernel Demo": page_3d_kernel_demo,
        "🎯 SVM Concepts": page_svm_concepts,
        "🤖 Model Training": page_model_training,
        "🔬 Hyperparameter Studio": page_hyperparameter_studio,
        "⚖️ Kernel Comparison": page_comparison_tool,
        "📄 Code Viewer": page_code_viewer,
    }

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "### 📚 Resources\n"
        "- [Scikit-learn SVM docs](https://scikit-learn.org/stable/modules/svm.html)\n"
        "- [RBF Kernel explanation](https://en.wikipedia.org/wiki/Radial_basis_function_kernel)\n"
        "- [Manim animations](../manim_animations/)"
    )

    page_func = page_map.get(page, page_home)
    page_func()


if __name__ == "__main__":
    main()
