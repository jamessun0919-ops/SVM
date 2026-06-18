"""
Phase 2: Real SVM Implementation Using Scikit-Learn
====================================================
Demonstrates actual machine learning implementation of RBF SVM.

Datasets: make_circles, make_moons, make_classification, Iris
Workflow: Generate -> Explore -> Split -> Scale -> Train -> Evaluate -> Visualize
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.datasets import make_circles, make_moons, make_classification, load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
)
import warnings

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

FIG_DIR = "figures"


def generate_datasets(random_state=42):
    datasets = {}

    X_c, y_c = make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=random_state)
    datasets["Circles"] = (X_c, y_c, "Concentric Circles")

    X_m, y_m = make_moons(n_samples=300, noise=0.1, random_state=random_state)
    datasets["Moons"] = (X_m, y_m, "Two Moons")

    X_clf, y_clf = make_classification(
        n_samples=300, n_features=2, n_redundant=0, n_informative=2,
        n_clusters_per_class=1, random_state=random_state,
    )
    datasets["Classification"] = (X_clf, y_clf, "Synthetic Classification")

    iris = load_iris()
    X_i = iris.data[:, :2]
    y_i = iris.target
    datasets["Iris"] = (X_i, y_i, "Iris (first 2 features)")

    return datasets


def plot_dataset(ax, X, y, title, cmap="bwr"):
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors="k", s=40, alpha=0.8)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    return scatter


def visualize_datasets(datasets):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for ax, (name, (X, y, desc)) in zip(axes, datasets.items()):
        plot_dataset(ax, X, y, desc)
        ax.legend(*ax.collections[0].legend_elements(), title="Class")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/datasets_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] Saved datasets_overview.png")


def plot_decision_surface(
    model, X, y, ax, title, cmap="bwr",
    alpha=0.3, support_vectors=True,
):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cmap_bg = ListedColormap(["#FFAAAA", "#AAAAFF"])
    ax.contourf(xx, yy, Z, alpha=alpha, cmap=cmap_bg)

    if support_vectors and hasattr(model, "support_vectors_"):
        sv = model.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=150, facecolors="none",
                   edgecolors="yellow", linewidths=2, label="Support Vectors")

    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors="k", s=40, alpha=0.8)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")


def train_evaluate_svm(X_train, X_test, y_train, y_test, C=1.0, gamma="scale"):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(kernel="rbf", C=C, gamma=gamma, probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1] if len(np.unique(y_train)) == 2 else None

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
        "F1 Score": f1_score(y_test, y_pred, average="weighted"),
    }

    if y_prob is not None:
        metrics["ROC AUC"] = roc_auc_score(y_test, y_prob)

    return model, scaler, metrics, y_pred, y_prob


def gamma_study(X_train, X_test, y_train, y_test):
    gamma_values = [0.001, 0.01, 0.1, 1, 10, 100]
    results = []
    models = {}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, (ax, gamma) in enumerate(zip(axes, gamma_values)):
        model, scaler, metrics, _, _ = train_evaluate_svm(
            X_train, X_test, y_train, y_test, C=1.0, gamma=gamma,
        )
        X_train_scaled = scaler.transform(X_train)
        plot_decision_surface(
            model, X_train_scaled, y_train, ax,
            f"γ = {gamma} | Acc = {metrics['Accuracy']:.3f}",
        )
        results.append({"gamma": gamma, **metrics})
        models[gamma] = model

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/gamma_study.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] Saved gamma_study.png")

    return pd.DataFrame(results)


def c_study(X_train, X_test, y_train, y_test):
    c_values = [0.1, 1, 10, 100, 1000]
    results = []
    models = {}

    fig, axes = plt.subplots(1, 5, figsize=(25, 4))

    for i, (ax, C) in enumerate(zip(axes, c_values)):
        model, scaler, metrics, _, _ = train_evaluate_svm(
            X_train, X_test, y_train, y_test, C=C, gamma="scale",
        )
        X_train_scaled = scaler.transform(X_train)
        plot_decision_surface(
            model, X_train_scaled, y_train, ax,
            f"C = {C} | Acc = {metrics['Accuracy']:.3f}",
        )
        results.append({"C": C, **metrics})
        models[C] = model

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/c_study.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] Saved c_study.png")

    return pd.DataFrame(results)


def plot_confusion_matrix(y_test, y_pred, title, filename):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Class 0", "Class 1"],
                yticklabels=["Class 0", "Class 1"])
    ax.set_title(f"Confusion Matrix - {title}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{filename}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {filename}")


def plot_roc_curve(y_test, y_prob, title, filename):
    if y_prob is None:
        return
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {title}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{filename}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {filename}")


def main():
    print("=" * 60)
    print("SVM RBF Implementation - Comprehensive Analysis")
    print("=" * 60)

    datasets = generate_datasets()
    print(f"\nGenerated {len(datasets)} datasets\n")

    visualize_datasets(datasets)

    dataset_name = "Circles"
    print(f"\n--- Focusing on: {dataset_name} ---\n")
    X, y, desc = datasets[dataset_name]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y,
    )
    print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")

    model, scaler, metrics, y_pred, y_prob = train_evaluate_svm(
        X_train, X_test, y_train, y_test, C=1.0, gamma="scale",
    )
    print("\n--- Base Model Performance ---")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    X_train_scaled = scaler.transform(X_train)
    plot_decision_surface(
        model, X_train_scaled, y_train, plt.gca(),
        f"RBF SVM on {desc}",
    )
    plt.savefig(f"{FIG_DIR}/circles_decision_boundary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] Saved circles_decision_boundary.png")

    plot_confusion_matrix(y_test, y_pred, dataset_name, "confusion_matrix.png")
    plot_roc_curve(y_test, y_prob, dataset_name, "roc_curve.png")

    print("\n--- Gamma Parameter Study ---")
    gamma_results = gamma_study(X_train, X_test, y_train, y_test)
    print(gamma_results.to_string(index=False))

    print("\n--- C Parameter Study ---")
    c_results = c_study(X_train, X_test, y_train, y_test)
    print(c_results.to_string(index=False))

    print("\n--- Cross-Validation (best gamma) ---")
    best_gamma = gamma_results.loc[gamma_results["Accuracy"].idxmax(), "gamma"]
    model_best = SVC(kernel="rbf", C=1.0, gamma=best_gamma, random_state=42)
    cv_scores = cross_val_score(model_best, X, y, cv=5)
    print(f"  Best gamma: {best_gamma}")
    print(f"  CV scores: {cv_scores}")
    print(f"  Mean CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    print("\n--- Additional Datasets ---")
    for name, (X_d, y_d, desc_d) in datasets.items():
        if name == dataset_name:
            continue
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_d, y_d, test_size=0.3, random_state=42, stratify=y_d,
        )
        if len(np.unique(y_d)) > 2:
            model_d = SVC(kernel="rbf", C=1.0, gamma="scale", decision_function_shape="ovo", random_state=42)
            scaler_d = StandardScaler()
            X_tr_scaled = scaler_d.fit_transform(X_tr)
            X_te_scaled = scaler_d.transform(X_te)
            model_d.fit(X_tr_scaled, y_tr)
            y_pr = model_d.predict(X_te_scaled)
            acc = accuracy_score(y_te, y_pr)
        else:
            _, _, m, _, _ = train_evaluate_svm(X_tr, X_te, y_tr, y_te)
            acc = m["Accuracy"]
        print(f"  {desc_d}: Accuracy = {acc:.4f}")

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
