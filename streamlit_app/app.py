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

st.set_page_config(
    page_title="SVM Kernel 3D Visualization Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

LANG = {
    "en": {
        "title": "SVM Kernel 3D Visualization Platform",
        "subtitle": "Explore how Support Vector Machines use Kernel Functions to transform non-linearly separable data into higher-dimensional space.",
        "sidebar_header": "## Controls\nAdjust dataset and model parameters below.",
        "dataset": "Dataset",
        "sample_size": "Sample Size",
        "noise": "Noise",
        "gamma": "Gamma (γ)",
        "c": "C (Regularization)",
        "kernel": "Kernel",
        "circles": "Circles",
        "moons": "Moons",
        "iris": "Iris",
        "nav_home": "🏠 Home",
        "nav_math": "📐 Math Foundation",
        "nav_3d": "🌐 3D Kernel Demo",
        "nav_concepts": "🎯 SVM Concepts",
        "nav_train": "🤖 Model Training",
        "nav_hparam": "🔬 Hyperparameter Studio",
        "nav_compare": "⚖️ Kernel Comparison",
        "nav_code": "📄 Code Viewer",
        "home_welcome": "Welcome to the SVM Kernel 3D Visualization Platform",
        "home_desc": "This interactive platform helps you understand how Support Vector Machines use Kernel Functions to transform data into higher-dimensional spaces where linear separation becomes possible.",
        "home_learning_path": "Learning Path:",
        "home_step1": "Mathematical Foundation — Understand the RBF kernel formula",
        "home_step2": "3D Kernel Demo — Watch data lift from 2D to 3D",
        "home_step3": "SVM Concepts — Learn hyperplane, margin, support vectors",
        "home_step4": "Model Training — Train real SVMs on your data",
        "home_step5": "Hyperparameter Studio — Explore gamma, C, kernel effects",
        "home_step6": "Comparison Tool — Compare kernels side by side",
        "home_card1_title": "Nonlinear Data",
        "home_card1_desc": "Circles & Moons<br>need kernel trick",
        "home_card2_title": "3D Feature Space",
        "home_card2_desc": "z = x² + y²<br>linear separability",
        "home_card3_title": "Kernel Trick",
        "home_card3_desc": "Compute implicitly<br>O(n) not O(n²)",
        "math_title": "RBF Kernel: Mathematical Foundation",
        "math_interactive_gamma": "Interactive γ",
        "math_dist_label": "Distance ||x - x'||",
        "math_sim_label": "Similarity K(x, x')",
        "math_chart_title": "RBF Kernel: Similarity vs Distance",
        "math_step1": "Euclidean Distance",
        "math_step2": "Squared Distance",
        "math_step3": "Scale by Gamma",
        "math_step4": "Exponential",
        "math_intuition_title": "Key Intuition",
        "math_intuition1": "When two points are <b>close</b> (distance ≈ 0), similarity ≈ 1",
        "math_intuition2": "When two points are <b>far</b> (distance → ∞), similarity → 0",
        "math_intuition3": "<b>γ</b> controls how quickly similarity drops off with distance",
        "math_intuition4": "<b>Small γ</b>: global influence, smooth decision boundary",
        "math_intuition5": "<b>Large γ</b>: local influence, complex decision boundary",
        "demo_title": "3D Kernel Demonstration",
        "demo_desc": "Watch how the kernel mapping z = x² + y² makes linearly inseparable data separable in 3D.",
        "demo_2d_title": "Original 2D Data",
        "demo_3d_title": "3D Feature Space",
        "demo_3d_desc": "Inner points → low z<br>Outer points → high z<br>A plane at z ≈ constant separates them!",
        "demo_learn_title": "Learning Points",
        "demo_learn1": "2D data is not linearly separable",
        "demo_learn2": "z = x² + y² lifts points into 3D",
        "demo_learn3": "In 3D, a flat plane separates classes",
        "demo_learn4": "The plane in 3D projects back to a circle in 2D",
        "demo_learn5": "This is the <b>kernel trick</b> in action!",
        "concept_title": "SVM Concepts",
        "concept_tab1": "Hyperplane",
        "concept_tab2": "Margin",
        "concept_tab3": "Support Vectors",
        "concept_tab4": "Kernel Trick",
        "hyperplane_title": "What is a Hyperplane?",
        "hyperplane_desc": "In 2D: A line. In 3D: A plane. In n-D: An (n-1)-dimensional subspace.",
        "hyperplane_w": "w = weight vector (normal to hyperplane)",
        "hyperplane_b": "b = bias term (offset from origin)",
        "hyperplane_decision": "Decision: sign(wᵀx + b) → +1 or -1",
        "hyperplane_chart": "No Single Line Separates Circles",
        "margin_title": "Maximum Margin Principle",
        "margin_desc": "SVM doesn't just find <i>any</i> separating hyperplane — it finds the one with the <b>maximum margin</b>.",
        "margin_point1": "Margin = distance from hyperplane to nearest points",
        "margin_point2": "Larger margin → better generalization",
        "margin_point3": "C controls how strictly we enforce the margin",
        "sv_title": "Support Vectors",
        "sv_desc": "Support Vectors are the training examples that lie on the margin boundary. They are the <b>only</b> points that determine the hyperplane!",
        "sv_point1": "Remove non-SVs → hyperplane unchanged",
        "sv_point2": "Remove one SV → hyperplane changes",
        "sv_point3": "SVM is sparse — only SVs matter",
        "kernel_trick_title": "The Kernel Trick",
        "kernel_trick_desc": "We can compute dot products in a high-dimensional feature space <b>without ever computing the mapping explicitly</b>.",
        "kernel_trick_infinite": "The RBF kernel corresponds to an <b>infinite-dimensional</b> feature space, yet we compute it in O(n) time!",
        "kernel_trick_kernels": "Available Kernels:",
        "train_title": "Train Your SVM Model",
        "train_accuracy": "Accuracy",
        "train_f1": "F1 Score",
        "train_model_info": "Model Info",
        "train_kernel": "Kernel",
        "train_gamma": "Gamma",
        "train_c": "C",
        "train_sv": "Support Vectors",
        "train_samples": "Training samples",
        "train_test": "Test samples",
        "train_cm": "Confusion Matrix",
        "train_predicted": "Predicted",
        "train_actual": "Actual",
        "train_report": "Classification Report",
        "hparam_title": "Hyperparameter Exploration Studio",
        "hparam_gamma_sweep": "Gamma Sweep (fixed C)",
        "hparam_c_sweep": "C Sweep (fixed γ)",
        "hparam_takeaways_title": "Key Takeaways",
        "hparam_takeaway1": "<b>γ too small</b> → Underfitting: boundary too smooth, low accuracy",
        "hparam_takeaway2": "<b>γ too large</b> → Overfitting: boundary too complex, isolated regions",
        "hparam_takeaway3": "<b>C too small</b> → Underfitting: margin too wide, many errors",
        "hparam_takeaway4": "<b>C too large</b> → Overfitting: margin too tight, sensitive to noise",
        "hparam_takeaway5": "<b>Goal</b>: Find the sweet spot where validation accuracy peaks",
        "compare_title": "Kernel Comparison Tool",
        "compare_perf": "Performance Comparison",
        "compare_guide_title": "When to Use Each Kernel",
        "compare_kernel": "Kernel",
        "compare_best_for": "Best For",
        "compare_boundary": "Decision Boundary",
        "compare_linear": "Linearly separable data, high-dimensional sparse data (text)",
        "compare_linear_b": "Straight line/plane",
        "compare_poly": "Data with polynomial relationships",
        "compare_poly_b": "Curved, polynomial shape",
        "compare_rbf": "Most nonlinear real-world data (default choice)",
        "compare_rbf_b": "Any smooth shape, universal approximator",
        "compare_sigmoid": "Neural network-like behavior",
        "compare_sigmoid_b": "Similar to RBF but less stable",
        "code_title": "Code Viewer",
        "code_desc": "View the Python implementation code for the SVM training pipeline.",
        "code_expander": "View Training Code",
        "resources": "Resources",
        "lang_label": "Language / 語言",
    },
    "zh": {
        "title": "SVM 核函數 3D 視覺化平台",
        "subtitle": "探索支援向量機如何使用核函數將非線性可分的資料轉換到更高維度的空間。",
        "sidebar_header": "## 控制面板\n調整下方資料集與模型參數。",
        "dataset": "資料集",
        "sample_size": "樣本數",
        "noise": "雜訊",
        "gamma": "Gamma (γ)",
        "c": "C (正規化)",
        "kernel": "核函數",
        "circles": "同心圓",
        "moons": "半月形",
        "iris": "鳶尾花",
        "nav_home": "🏠 首頁",
        "nav_math": "📐 數學基礎",
        "nav_3d": "🌐 3D 核函數展示",
        "nav_concepts": "🎯 SVM 概念",
        "nav_train": "🤖 模型訓練",
        "nav_hparam": "🔬 超參數探索",
        "nav_compare": "⚖️ 核函數比較",
        "nav_code": "📄 程式碼檢視",
        "home_welcome": "歡迎使用 SVM 核函數 3D 視覺化平台",
        "home_desc": "這個互動平台幫助你理解支援向量機如何利用核函數將資料轉換到更高維度的空間，使線性分割成為可能。",
        "home_learning_path": "學習路徑：",
        "home_step1": "數學基礎 — 理解 RBF 核函數公式",
        "home_step2": "3D 核函數展示 — 觀看資料從 2D 提升到 3D",
        "home_step3": "SVM 概念 — 學習超平面、邊界、支援向量",
        "home_step4": "模型訓練 — 在真實資料上訓練 SVM",
        "home_step5": "超參數探索 — 探索 gamma、C、核函數效果",
        "home_step6": "比較工具 — 並排比較不同核函數",
        "home_card1_title": "非線性資料",
        "home_card1_desc": "同心圓與半月形<br>需要核函數技巧",
        "home_card2_title": "3D 特徵空間",
        "home_card2_desc": "z = x² + y²<br>產生線性可分性",
        "home_card3_title": "核函數技巧",
        "home_card3_desc": "隱式計算<br>O(n) 而非 O(n²)",
        "math_title": "RBF 核函數：數學基礎",
        "math_interactive_gamma": "互動式 γ",
        "math_dist_label": "距離 ||x - x'||",
        "math_sim_label": "相似度 K(x, x')",
        "math_chart_title": "RBF 核函數：相似度 vs 距離",
        "math_step1": "歐幾里德距離",
        "math_step2": "平方距離",
        "math_step3": "Gamma 縮放",
        "math_step4": "指數函數",
        "math_intuition_title": "關鍵直覺",
        "math_intuition1": "當兩點 <b>靠近</b>（距離 ≈ 0），相似度 ≈ 1",
        "math_intuition2": "當兩點 <b>遠離</b>（距離 → ∞），相似度 → 0",
        "math_intuition3": "<b>γ</b> 控制相似度隨距離下降的速度",
        "math_intuition4": "<b>小 γ</b>：全域影響，平滑決策邊界",
        "math_intuition5": "<b>大 γ</b>：局部影響，複雜決策邊界",
        "demo_title": "3D 核函數展示",
        "demo_desc": "觀看核映射 z = x² + y² 如何讓線性不可分的資料在 3D 中變得可分。",
        "demo_2d_title": "原始 2D 資料",
        "demo_3d_title": "3D 特徵空間",
        "demo_3d_desc": "內部點 → 低 z<br>外部點 → 高 z<br>平面在 z ≈ 常數處分隔它們！",
        "demo_learn_title": "學習要點",
        "demo_learn1": "2D 資料無法線性分割",
        "demo_learn2": "z = x² + y² 將點提升到 3D",
        "demo_learn3": "在 3D 中，一個平面就能分隔類別",
        "demo_learn4": "3D 中的平面投影回 2D 變成圓形",
        "demo_learn5": "這就是 <b>核函數技巧</b> 的實際展現！",
        "concept_title": "SVM 概念",
        "concept_tab1": "超平面",
        "concept_tab2": "邊界",
        "concept_tab3": "支援向量",
        "concept_tab4": "核函數技巧",
        "hyperplane_title": "什麼是超平面？",
        "hyperplane_desc": "在 2D：一條線。在 3D：一個平面。在 n-D：一個 (n-1) 維子空間。",
        "hyperplane_w": "w = 權重向量（超平面的法向量）",
        "hyperplane_b": "b = 偏置項（偏移原點）",
        "hyperplane_decision": "決策：sign(wᵀx + b) → +1 或 -1",
        "hyperplane_chart": "沒有直線能分開同心圓",
        "margin_title": "最大邊界原則",
        "margin_desc": "SVM 不只是找到 <i>任何</i> 分隔超平面——它找到具有 <b>最大邊界</b> 的那個。",
        "margin_point1": "邊界 = 從超平面到最近點的距離",
        "margin_point2": "較大邊界 → 更好的泛化能力",
        "margin_point3": "C 控制我們對邊界要求的嚴格程度",
        "sv_title": "支援向量",
        "sv_desc": "支援向量是位於邊界上的訓練樣本。它們是 <b>唯一</b> 決定超平面的點！",
        "sv_point1": "移除非支援向量 → 超平面不變",
        "sv_point2": "移除一個支援向量 → 超平面改變",
        "sv_point3": "SVM 是稀疏的——只有支援向量重要",
        "kernel_trick_title": "核函數技巧",
        "kernel_trick_desc": "我們可以在高維特徵空間中計算內積，<b>而無需顯式計算映射</b>。",
        "kernel_trick_infinite": "RBF 核函數對應到一個 <b>無限維</b> 的特徵空間，但我們只需 O(n) 時間就能計算！",
        "kernel_trick_kernels": "可用的核函數：",
        "train_title": "訓練你的 SVM 模型",
        "train_accuracy": "準確率",
        "train_f1": "F1 分數",
        "train_model_info": "模型資訊",
        "train_kernel": "核函數",
        "train_gamma": "Gamma",
        "train_c": "C",
        "train_sv": "支援向量",
        "train_samples": "訓練樣本",
        "train_test": "測試樣本",
        "train_cm": "混淆矩陣",
        "train_predicted": "預測",
        "train_actual": "實際",
        "train_report": "分類報告",
        "hparam_title": "超參數探索工作室",
        "hparam_gamma_sweep": "Gamma 掃描（固定 C）",
        "hparam_c_sweep": "C 掃描（固定 γ）",
        "hparam_takeaways_title": "重點整理",
        "hparam_takeaway1": "<b>γ 太小</b> → 欠擬合：邊界過於平滑，準確率低",
        "hparam_takeaway2": "<b>γ 太大</b> → 過擬合：邊界過於複雜，產生孤立區域",
        "hparam_takeaway3": "<b>C 太小</b> → 欠擬合：邊界過寬，錯誤較多",
        "hparam_takeaway4": "<b>C 太大</b> → 過擬合：邊界過窄，對雜訊敏感",
        "hparam_takeaway5": "<b>目標</b>：找到驗證準確率最高的最佳點",
        "compare_title": "核函數比較工具",
        "compare_perf": "效能比較",
        "compare_guide_title": "各核函數的使用時機",
        "compare_kernel": "核函數",
        "compare_best_for": "最佳用途",
        "compare_boundary": "決策邊界",
        "compare_linear": "線性可分資料、高維稀疏資料（文字）",
        "compare_linear_b": "直線／平面",
        "compare_poly": "具有多項式關係的資料",
        "compare_poly_b": "彎曲的多項式形狀",
        "compare_rbf": "大多數非線性真實世界資料（預設選擇）",
        "compare_rbf_b": "任意平滑形狀，通用逼近器",
        "compare_sigmoid": "類似神經網路的行為",
        "compare_sigmoid_b": "類似 RBF 但較不穩定",
        "code_title": "程式碼檢視",
        "code_desc": "檢視 SVM 訓練流程的 Python 實作程式碼。",
        "code_expander": "檢視訓練程式碼",
        "resources": "參考資源",
        "lang_label": "Language / 語言",
    },
}


def T(key):
    lang = st.session_state.get("lang", "en")
    return LANG[lang].get(key, LANG["en"].get(key, key))


st.markdown(
    """
<style>
    .main-header { font-size: 2.2rem; font-weight: 800; color: #ffffff;
                   text-align: center; padding: 1.2rem 1rem;
                   background: linear-gradient(135deg, #1a1a2e, #16213e);
                   border-radius: 12px; margin-bottom: 0.5rem;
                   box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
    .sub-header { font-size: 1.5rem; font-weight: 700; color: #00B4D8;
                  padding: 0.5rem 0; border-bottom: 3px solid #00B4D8;
                  margin-bottom: 0.5rem; }
    .card { background: #ffffff; border-radius: 10px; padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: 0.5rem 0;
            color: #1a1a2e; border: 1px solid #e0e0e0; }
    .card h3, .card h4, .card p, .card li, .card b, .card td, .card th { color: #1a1a2e; }
    .metric-box { background: linear-gradient(135deg, #667eea, #764ba2);
                  color: white; border-radius: 8px; padding: 1rem;
                  text-align: center; }
    .metric-box h2, .metric-box h4, .metric-box p { color: white; margin: 0; }
    .formula { font-family: 'Courier New', monospace;
               background: #2d2d2d; color: #f8f8f2; padding: 0.8rem;
               border-radius: 5px; font-size: 1.1rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(f'<div class="main-header">🧠 {T("title")}</div>', unsafe_allow_html=True)
st.markdown(T("subtitle"))

SIDEBAR_HEADER = T("sidebar_header")


def dataset_en_name(name):
    mapping = {"Circles": "Circles", "\u540c\u5fc3\u5713": "Circles",
               "Moons": "Moons", "\u534a\u6708\u5f62": "Moons",
               "Iris": "Iris", "\u9ce9\u5c3e\u82b1": "Iris"}
    return mapping.get(name, name)

def generate_dataset(name, n_samples, noise, random_state=42):
    if name in ("Circles", "同心圓"):
        X, y = make_circles(n_samples=n_samples, noise=noise,
                            factor=0.5, random_state=random_state)
    elif name in ("Moons", "半月形"):
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    elif name in ("Iris", "鳶尾花"):
        iris = load_iris()
        X = iris.data[:, :2]
        y = iris.target
    else:
        X, y = make_circles(n_samples=n_samples, noise=noise,
                            random_state=random_state)
    return X, y


def compute_kernel_z(X, kernel):
    x, y = X[:, 0], X[:, 1]
    if kernel == "linear":
        return x + y
    elif kernel == "poly":
        return x**2 + x * y + y**2
    elif kernel == "sigmoid":
        return np.tanh(x + y)
    else:
        return x**2 + y**2


def kernel_formula_str(kernel):
    mapping = {
        "rbf": "z = x\u00b2 + y\u00b2",
        "linear": "z = x + y",
        "poly": "z = x\u00b2 + xy + y\u00b2",
        "sigmoid": "z = tanh(x + y)",
    }
    return mapping.get(kernel, "z = x\u00b2 + y\u00b2")


def plotly_3d_kernel_mapping(X, y, gamma_val, kernel="rbf"):
    z = compute_kernel_z(X, kernel)
    fig = go.Figure()
    colors = {0: "blue", 1: "red"}

    pad = 0.15
    x_pad = (X[:, 0].max() - X[:, 0].min()) * pad
    y_pad = (X[:, 1].max() - X[:, 1].min()) * pad
    xs = np.linspace(X[:, 0].min() - x_pad, X[:, 0].max() + x_pad, 30)
    ys = np.linspace(X[:, 1].min() - y_pad, X[:, 1].max() + y_pad, 30)
    xx, yy = np.meshgrid(xs, ys)
    zz_surf = compute_kernel_z(np.column_stack([xx.ravel(), yy.ravel()]), kernel).reshape(xx.shape)

    fig.add_trace(go.Surface(
        x=xx, y=yy, z=zz_surf,
        colorscale=[[0, "#FF8C00"], [0.5, "#FFD700"], [1, "#FF8C00"]],
        opacity=0.3, showscale=False, name="Kernel Surface",
    ))

    for class_id in [0, 1]:
        mask = y == class_id
        fig.add_trace(go.Scatter3d(
            x=X[mask, 0], y=X[mask, 1], z=z[mask],
            mode="markers",
            marker=dict(size=5, color=colors[class_id], opacity=0.9, line=dict(width=0.5, color="black")),
            name=f"Class {'A' if class_id == 0 else 'B'}",
        ))

    z_mean = z.mean()
    xx_plane, yy_plane = np.meshgrid(
        np.linspace(X[:, 0].min() - x_pad, X[:, 0].max() + x_pad, 20),
        np.linspace(X[:, 1].min() - y_pad, X[:, 1].max() + y_pad, 20),
    )
    zz_plane = np.full_like(xx_plane, z_mean)
    fig.add_trace(go.Surface(
        x=xx_plane, y=yy_plane, z=zz_plane,
        colorscale=[[0, "green"], [1, "green"]],
        opacity=0.15, showscale=False, name="Hyperplane",
    ))

    formula = kernel_formula_str(kernel)
    fig.update_layout(
        title=f"Kernel Mapping: {formula} (\u03b3 = {gamma_val})",
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
        ),
        height=600,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def plotly_kernel_lift_animation(X, y, kernel, z_target):
    n_frames = 30
    colors = {0: "blue", 1: "red"}

    x_pad = (X[:, 0].max() - X[:, 0].min()) * 0.15
    y_pad = (X[:, 1].max() - X[:, 1].min()) * 0.15
    xs = np.linspace(X[:, 0].min() - x_pad, X[:, 0].max() + x_pad, 25)
    ys = np.linspace(X[:, 1].min() - y_pad, X[:, 1].max() + y_pad, 25)
    xx, yy = np.meshgrid(xs, ys)
    zz_surf = compute_kernel_z(np.column_stack([xx.ravel(), yy.ravel()]), kernel).reshape(xx.shape)

    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=xx, y=yy, z=zz_surf,
        colorscale=[[0, "#FF8C00"], [0.5, "#FFD700"], [1, "#FF8C00"]],
        opacity=0.3, showscale=False, name="Kernel Surface",
    ))

    for class_id in [0, 1]:
        mask = y == class_id
        fig.add_trace(go.Scatter3d(
            x=X[mask, 0], y=X[mask, 1], z=np.zeros(mask.sum()),
            mode="markers",
            marker=dict(size=5, color=colors[class_id], opacity=0.9,
                        line=dict(width=0.5, color="black")),
            name=f"Class {'A' if class_id == 0 else 'B'}",
        ))

    z_mean = z_target.mean()
    zz_plane = np.full_like(xx, z_mean)
    fig.add_trace(go.Surface(
        x=xx, y=yy, z=zz_plane,
        colorscale=[[0, "green"], [1, "green"]],
        opacity=0.15, showscale=False, name="Hyperplane",
    ))

    formula = kernel_formula_str(kernel)
    frames = []
    cam_start = dict(eye=dict(x=0.01, y=0.01, z=2.5))
    cam_end = dict(eye=dict(x=1.8, y=1.8, z=0.8))

    for i in range(n_frames):
        t = i / (n_frames - 1)
        t_smooth = t * t * (3 - 2 * t)
        eye_x = cam_start["eye"]["x"] + (cam_end["eye"]["x"] - cam_start["eye"]["x"]) * t_smooth
        eye_y = cam_start["eye"]["y"] + (cam_end["eye"]["y"] - cam_start["eye"]["y"]) * t_smooth
        eye_z = cam_start["eye"]["z"] + (cam_end["eye"]["z"] - cam_start["eye"]["z"]) * t_smooth

        z_anim = z_target * t_smooth
        idx = 0
        data = []
        data.append(dict(type="surface", x=xx, y=yy, z=zz_surf,
                         colorscale=[[0, "#FF8C00"], [0.5, "#FFD700"], [1, "#FF8C00"]],
                         opacity=0.3, showscale=False))
        for class_id in [0, 1]:
            mask = y == class_id
            data.append(dict(type="scatter3d",
                             x=X[mask, 0], y=X[mask, 1], z=z_anim[mask],
                             mode="markers",
                             marker=dict(size=5, color=colors[class_id], opacity=0.9,
                                        line=dict(width=0.5, color="black")),
                             name=f"Class {'A' if class_id == 0 else 'B'}"))
        data.append(dict(type="surface", x=xx, y=yy, z=zz_plane,
                         colorscale=[[0, "green"], [1, "green"]],
                         opacity=0.15, showscale=False))

        frames.append(go.Frame(
            data=data,
            layout=dict(scene=dict(camera=dict(eye=dict(x=eye_x, y=eye_y, z=eye_z)))),
            name=f"frame{i}",
        ))

    fig.frames = frames
    fig.update_layout(
        title=f"2D → 3D Lift: {formula}",
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            camera=dict(eye=dict(x=0.01, y=0.01, z=2.5)),
        ),
        height=500,
        margin=dict(l=0, r=0, t=40, b=0),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            x=0.5, y=-0.05, xanchor="center",
            buttons=[dict(
                label="Play",
                method="animate",
                args=[None, dict(frame=dict(duration=80, redraw=True),
                                 fromcurrent=True, transition=dict(duration=0))],
            ), dict(
                label="Reset",
                method="animate",
                args=[[None], dict(frame=dict(duration=0, redraw=True),
                                   fromcurrent=True, transition=dict(duration=0))],
            )],
        )],
        sliders=[dict(
            active=0,
            x=0.5, y=-0.12, xanchor="center", len=0.8,
            steps=[dict(
                method="animate",
                args=[[f"frame{i}"],
                      dict(mode="immediate", frame=dict(duration=0, redraw=True),
                           transition=dict(duration=0))],
                label=f"{i*100//(n_frames-1)}%",
            ) for i in range(0, n_frames, 3)],
        )],
    )
    return fig


def plot_decision_boundary(model, X, y, scaler, title, ax):
    X_scaled = scaler.transform(X)
    x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
    y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    n_classes = len(np.unique(y))
    if n_classes <= 2:
        cmap_bg = ListedColormap(["#FFAAAA", "#AAAACC"])
    else:
        cmap_bg = ListedColormap(["#FFAAAA", "#AAAACC", "#AAFFAA"])
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
    lang = st.session_state.get("lang", "en")
    st.sidebar.markdown(SIDEBAR_HEADER)
    dataset_options = [T("circles"), T("moons"), T("iris")]
    dataset = st.sidebar.selectbox(T("dataset"), dataset_options)
    n_samples = st.sidebar.slider(T("sample_size"), 50, 1000, 300, 50)
    noise = st.sidebar.slider(T("noise"), 0.0, 0.5, 0.1, 0.05)
    gamma = st.sidebar.slider(T("gamma"), 0.001, 100.0, 1.0, 0.1, format="%.3f")
    C = st.sidebar.slider(T("c"), 0.01, 1000.0, 1.0, 0.1, format="%.2f")
    kernel = st.sidebar.selectbox(T("kernel"), ["rbf", "linear", "poly", "sigmoid"])
    return dataset, n_samples, noise, gamma, C, kernel


def page_home():
    st.markdown(f"""
    <div class="card">
    <h3>{T("home_welcome")}</h3>
    <p>{T("home_desc")}</p>
    <h4>{T("home_learning_path")}</h4>
    <ol>
        <li>{T("home_step1")}</li>
        <li>{T("home_step2")}</li>
        <li>{T("home_step3")}</li>
        <li>{T("home_step4")}</li>
        <li>{T("home_step5")}</li>
        <li>{T("home_step6")}</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-box">
        <h4>{T("home_card1_title")}</h4>
        <p>{T("home_card1_desc")}</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-box">
        <h4>{T("home_card2_title")}</h4>
        <p>{T("home_card2_desc")}</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-box">
        <h4>{T("home_card3_title")}</h4>
        <p>{T("home_card3_desc")}</p>
        </div>
        """, unsafe_allow_html=True)


def page_mathematical_foundation():
    st.markdown(f'<div class="sub-header">{T("math_title")}</div>', unsafe_allow_html=True)

    gamma_val = st.slider(T("math_interactive_gamma"), 0.1, 5.0, 1.0, 0.1)
    dist = np.linspace(0, 5, 100)
    similarity = np.exp(-gamma_val * dist ** 2)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dist, similarity, "b-", linewidth=3, label=f"\u03b3 = {gamma_val}")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Distance ||x - x'||", fontsize=12)
    ax.set_ylabel("Similarity K(x, x')", fontsize=12)
    ax.set_title("RBF Kernel: Similarity vs Distance", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)
    plt.close()

    steps = [
        (T("math_step1"), r"d(x, x') = \|x - x'\|"),
        (T("math_step2"), r"\|x - x'\|^2"),
        (T("math_step3"), r"\gamma\|x - x'\|^2"),
        (T("math_step4"), r"\exp(-\gamma\|x - x'\|^2)"),
    ]
    cols = st.columns(4)
    for i, (title, formula) in enumerate(steps):
        with cols[i]:
            st.markdown(f"**{i+1}. {title}**")
            st.markdown(f"<div class='formula'>$${formula}$$</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card">
    <h4>{T("math_intuition_title")}</h4>
    <ul>
        <li>{T("math_intuition1")}</li>
        <li>{T("math_intuition2")}</li>
        <li>{T("math_intuition3")}</li>
        <li>{T("math_intuition4")}</li>
        <li>{T("math_intuition5")}</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


def page_3d_kernel_demo():
    st.markdown(f'<div class="sub-header">{T("demo_title")}</div>', unsafe_allow_html=True)
    st.markdown(T("demo_desc"))

    dataset, n_samples, noise, gamma, C, kernel = sidebar_controls()
    X, y = generate_dataset(dataset, n_samples, noise)
    if dataset in ("Iris", "鳶尾花"):
        y = (y > 0).astype(int)

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown(f"### {T('demo_2d_title')}")
        fig2d, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k", s=40)
        ax.set_title(f"{dataset_en_name(dataset)} (2D)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        st.pyplot(fig2d)
        plt.close()

    with col_right:
        st.markdown(f"### {T('demo_3d_title')}")
        fig = plotly_3d_kernel_mapping(X, y, gamma, kernel)
        st.plotly_chart(fig, use_container_width=True)

    z = compute_kernel_z(X, kernel)
    formula = kernel_formula_str(kernel)
    st.markdown(f"""
    <div class="card">
    <b>{formula}</b><br>
    {T("demo_3d_desc")}
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="card">
    <h4>{T("demo_learn_title")}</h4>
    <ul>
        <li>{T("demo_learn1")}</li>
        <li>{T("demo_learn2")}</li>
        <li>{T("demo_learn3")}</li>
        <li>{T("demo_learn4")}</li>
        <li>{T("demo_learn5")}</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 2D → 3D Lift Animation")
    fig_anim = plotly_kernel_lift_animation(X, y, kernel, z)
    st.plotly_chart(fig_anim, use_container_width=True)


def page_svm_concepts():
    st.markdown(f'<div class="sub-header">{T("concept_title")}</div>', unsafe_allow_html=True)

    tabs = st.tabs([T("concept_tab1"), T("concept_tab2"), T("concept_tab3"), T("concept_tab4")])

    with tabs[0]:
        st.markdown(f"""
        <div class="card">
        <h3>{T("hyperplane_title")}</h3>
        <p>{T("hyperplane_desc")}</p>
        <div class="formula">
        $$w^T x + b = 0$$
        </div>
        <ul>
            <li>{T("hyperplane_w")}</li>
            <li>{T("hyperplane_b")}</li>
            <li>{T("hyperplane_decision")}</li>
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
            ax.plot(x_vals, y_vals, "--", alpha=0.3, label=f"\u03b8={angle}\u00b0")
        ax.set_title("No Single Line Separates Circles", fontsize=14, fontweight="bold")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        st.pyplot(fig)
        plt.close()

    with tabs[1]:
        st.markdown(f"""
        <div class="card">
        <h3>{T("margin_title")}</h3>
        <p>{T("margin_desc")}</p>
        <div class="formula">
        $$\\max_{{w,b}} \\frac{{2}}{{\\|w\\|}} \\quad \\text{{s.t.}} \\quad y_i(w^T x_i + b) \\geq 1$$
        </div>
        <ul>
            <li>{T("margin_point1")}</li>
            <li>{T("margin_point2")}</li>
            <li>{T("margin_point3")}</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with tabs[2]:
        st.markdown(f"""
        <div class="card">
        <h3>{T("sv_title")}</h3>
        <p>{T("sv_desc")}</p>
        <ul>
            <li>{T("sv_point1")}</li>
            <li>{T("sv_point2")}</li>
            <li>{T("sv_point3")}</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with tabs[3]:
        st.markdown(f"""
        <div class="card">
        <h3>{T("kernel_trick_title")}</h3>
        <p>{T("kernel_trick_desc")}</p>
        <div class="formula">
        $$K(x, x') = \\phi(x)^T \\phi(x')$$
        </div>
        <p>{T("kernel_trick_infinite")}</p>
        <h4>{T("kernel_trick_kernels")}</h4>
        <ul>
            <li><b>Linear:</b> K(x,x') = x\u1d40x'</li>
            <li><b>Polynomial:</b> K(x,x') = (\u03b3x\u1d40x' + r)^d</li>
            <li><b>RBF:</b> K(x,x') = exp(-\u03b3||x - x'||\u00b2)</li>
            <li><b>Sigmoid:</b> K(x,x') = tanh(\u03b3x\u1d40x' + r)</li>
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
    st.markdown(f'<div class="sub-header">{T("train_title")}</div>', unsafe_allow_html=True)

    dataset, n_samples, noise, gamma, C, kernel = sidebar_controls()
    X, y = generate_dataset(dataset, n_samples, noise)
    if dataset in ("Iris", "鳶尾花"):
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
                               f"SVM ({kernel}) on {dataset_en_name(dataset)}", ax)
        st.pyplot(fig)
        plt.close()

    with col2:
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        st.markdown(f"""
        <div class="metric-box">
            <h2>{acc:.3f}</h2>
            <p>{T("train_accuracy")}</p>
        </div>
        <br>
        <div class="metric-box">
            <h2>{f1:.3f}</h2>
            <p>{T("train_f1")}</p>
        </div>
        <br>
        <div class="card">
            <b>{T("train_model_info")}</b><br>
            {T("train_kernel")}: {kernel}<br>
            {T("train_gamma")}: {gamma}<br>
            {T("train_c")}: {C}<br>
            {T("train_sv")}: {len(model.support_vectors_)}<br>
            {T("train_samples")}: {len(X_train)}<br>
            {T("train_test")}: {len(X_test)}
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"### {T('train_cm')}")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    ax_cm.imshow(cm, interpolation="nearest", cmap="Blues")
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xlabel("Predicted", fontsize=12)
    ax_cm.set_ylabel("Actual", fontsize=12)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16)
    st.pyplot(fig_cm)
    plt.close()

    with st.expander(T("train_report")):
        st.text(classification_report(y_test, y_pred))


def page_hyperparameter_studio():
    st.markdown(f'<div class="sub-header">{T("hparam_title")}</div>', unsafe_allow_html=True)

    dataset, n_samples, noise, gamma, C, kernel = sidebar_controls()
    X, y = generate_dataset(dataset, n_samples, noise)
    if dataset in ("Iris", "鳶尾花"):
        y = (y > 0).astype(int)

    st.markdown(f"### {T('hparam_gamma_sweep')}")
    gamma_values = [0.001, 0.01, 0.1, 1, 10, 100]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, (ax, g) in enumerate(zip(axes, gamma_values)):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        m, s = train_svm(X_tr, y_tr, kernel, g, C)
        y_pr = m.predict(s.transform(X_te))
        acc = accuracy_score(y_te, y_pr)
        plot_decision_boundary(m, X_tr, y_tr, s, f"\u03b3={g} (acc={acc:.3f})", ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown(f"### {T('hparam_c_sweep')}")
    c_values = [0.01, 0.1, 1, 10, 100, 1000]
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
    axes2 = axes2.flatten()
    for i, (ax, c_val) in enumerate(zip(axes2, c_values)):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        m, s = train_svm(X_tr, y_tr, kernel, gamma, c_val)
        y_pr = m.predict(s.transform(X_te))
        acc = accuracy_score(y_te, y_pr)
        plot_decision_boundary(m, X_tr, y_tr, s, f"C={c_val} (acc={acc:.3f})", ax)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.markdown(f"""
    <div class="card">
    <h4>{T("hparam_takeaways_title")}</h4>
    <ul>
        <li>{T("hparam_takeaway1")}</li>
        <li>{T("hparam_takeaway2")}</li>
        <li>{T("hparam_takeaway3")}</li>
        <li>{T("hparam_takeaway4")}</li>
        <li>{T("hparam_takeaway5")}</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


def page_comparison_tool():
    st.markdown(f'<div class="sub-header">{T("compare_title")}</div>', unsafe_allow_html=True)

    dataset, n_samples, noise, gamma, C, _ = sidebar_controls()
    X, y = generate_dataset(dataset, n_samples, noise)
    if dataset in ("Iris", "鳶尾花"):
        y = (y > 0).astype(int)

    kernels = ["linear", "poly", "rbf", "sigmoid"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    results = []
    for ax, kernel in zip(axes, kernels):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        m, s = train_svm(X_tr, y_tr, kernel, gamma, C)
        y_pr = m.predict(s.transform(X_te))
        acc = accuracy_score(y_te, y_pr)
        n_sv = len(m.support_vectors_) if hasattr(m, "support_vectors_") else 0
        results.append({"Kernel": kernel, "Accuracy": f"{acc:.4f}", "Support Vectors": n_sv})
        plot_decision_boundary(m, X_tr, y_tr, s, f"{kernel.upper()} (acc={acc:.4f})", ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown(f"### {T('compare_perf')}")
    st.dataframe(pd.DataFrame(results), use_container_width=True)

    st.markdown(f"""
    <div class="card">
    <h4>{T("compare_guide_title")}</h4>
    <table>
        <tr><th>{T("compare_kernel")}</th><th>{T("compare_best_for")}</th><th>{T("compare_boundary")}</th></tr>
        <tr><td><b>Linear</b></td><td>{T("compare_linear")}</td><td>{T("compare_linear_b")}</td></tr>
        <tr><td><b>Polynomial</b></td><td>{T("compare_poly")}</td><td>{T("compare_poly_b")}</td></tr>
        <tr><td><b>RBF</b></td><td>{T("compare_rbf")}</td><td>{T("compare_rbf_b")}</td></tr>
        <tr><td><b>Sigmoid</b></td><td>{T("compare_sigmoid")}</td><td>{T("compare_sigmoid_b")}</td></tr>
    </table>
    </div>
    """, unsafe_allow_html=True)


def page_code_viewer():
    st.markdown(f'<div class="sub-header">{T("code_title")}</div>', unsafe_allow_html=True)
    st.markdown(T("code_desc"))
    with st.expander(T("code_expander"), expanded=True):
        st.code("""
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def train_rbf_svm(X, y, C=1.0, gamma='scale'):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y,
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = SVC(
        kernel='rbf', C=C, gamma=gamma,
        probability=True, random_state=42,
    )
    model.fit(X_train_scaled, y_train)
    accuracy = model.score(X_test_scaled, y_test)
    return model, scaler, accuracy
        """, language="python")


def main():
    if "lang" not in st.session_state:
        st.session_state.lang = "en"

    st.sidebar.markdown("---")
    col_l, col_r = st.sidebar.columns([1, 1])
    with col_l:
        if st.button("English", use_container_width=True,
                     type="primary" if st.session_state.lang == "en" else "secondary"):
            st.session_state.lang = "en"
            st.rerun()
    with col_r:
        if st.button("中文", use_container_width=True,
                     type="primary" if st.session_state.lang == "zh" else "secondary"):
            st.session_state.lang = "zh"
            st.rerun()

    page = st.sidebar.radio(
        "",
        [
            T("nav_home"),
            T("nav_math"),
            T("nav_3d"),
            T("nav_concepts"),
            T("nav_train"),
            T("nav_hparam"),
            T("nav_compare"),
            T("nav_code"),
        ],
    )

    page_map = {
        T("nav_home"): page_home,
        T("nav_math"): page_mathematical_foundation,
        T("nav_3d"): page_3d_kernel_demo,
        T("nav_concepts"): page_svm_concepts,
        T("nav_train"): page_model_training,
        T("nav_hparam"): page_hyperparameter_studio,
        T("nav_compare"): page_comparison_tool,
        T("nav_code"): page_code_viewer,
    }

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"### {T('resources')}\n"
        "- [Scikit-learn SVM docs](https://scikit-learn.org/stable/modules/svm.html)\n"
        "- [RBF Kernel explanation](https://en.wikipedia.org/wiki/Radial_basis_function_kernel)\n"
    )

    page_func = page_map.get(page, page_home)
    page_func()


if __name__ == "__main__":
    main()
