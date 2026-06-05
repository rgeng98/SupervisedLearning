"""
KMeans Clustering with Train/Validation Split
==============================================
Fits KMeans on training data, evaluates on both train and validation sets,
and selects the optimal k via the Elbow method and Silhouette score.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.decomposition import PCA


# ============================================================================
# 1. Data preparation
# ============================================================================

def load_data(
    n_samples: int = 1500,
    n_features: int = 8,
    n_true_clusters: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate (or replace with your own) dataset.
    Returns X_train, X_val as scaled numpy arrays.
    """
    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_true_clusters,
        cluster_std=1.2,
        random_state=random_state,
    )

    X_train, X_val = train_test_split(X, test_size=0.2, random_state=random_state)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)   # fit only on train
    X_val   = scaler.transform(X_val)         # apply same transform to val

    print(f"Train size : {X_train.shape}")
    print(f"Val size   : {X_val.shape}")
    return X_train, X_val


# ============================================================================
# 2. Evaluation helpers
# ============================================================================

def evaluate(X: np.ndarray, labels: np.ndarray, split: str) -> dict:
    """Compute clustering metrics. Requires at least 2 distinct labels."""
    if len(set(labels)) < 2:
        return {}
    metrics = {
        "split":             split,
        "inertia":           None,          # filled by caller
        "silhouette":        silhouette_score(X, labels),
        "calinski_harabasz": calinski_harabasz_score(X, labels),
        "davies_bouldin":    davies_bouldin_score(X, labels),
    }
    return metrics


# ============================================================================
# 3. Optimal-k search
# ============================================================================

def find_optimal_k(
    X_train: np.ndarray,
    X_val: np.ndarray,
    k_range: range = range(2, 12),
    random_state: int = 42,
) -> tuple[int, list[dict]]:
    """
    Fit KMeans for each k and collect metrics on both splits.
    Returns the best k (by validation silhouette) and full results.
    """
    results = []

    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10,
                    max_iter=300, random_state=random_state)
        km.fit(X_train)

        train_labels = km.labels_
        val_labels   = km.predict(X_val)

        train_metrics = evaluate(X_train, train_labels, "train")
        val_metrics   = evaluate(X_val,   val_labels,   "val")

        train_metrics["inertia"] = km.inertia_
        val_metrics["inertia"]   = float(
            np.sum(np.min(
                np.linalg.norm(X_val[:, None] - km.cluster_centers_[None], axis=2) ** 2,
                axis=1
            ))
        )

        results.append({
            "k":      k,
            "model":  km,
            "train":  train_metrics,
            "val":    val_metrics,
        })

        print(
            f"  k={k:2d} | "
            f"train sil={train_metrics['silhouette']:.4f}  "
            f"val sil={val_metrics['silhouette']:.4f}  "
            f"inertia={km.inertia_:.1f}"
        )

    # Best k: highest validation silhouette
    best = max(results, key=lambda r: r["val"]["silhouette"])
    print(f"\nBest k = {best['k']}  (val silhouette = {best['val']['silhouette']:.4f})")
    return best["k"], results


# ============================================================================
# 4. Final model
# ============================================================================

def fit_final_model(
    X_train: np.ndarray,
    X_val: np.ndarray,
    k: int,
    random_state: int = 42,
) -> tuple[KMeans, np.ndarray, np.ndarray]:
    """Fit final KMeans with chosen k and return model + label arrays."""
    km = KMeans(n_clusters=k, init="k-means++", n_init=20,
                max_iter=500, random_state=random_state)
    km.fit(X_train)

    train_labels = km.labels_
    val_labels   = km.predict(X_val)
    return km, train_labels, val_labels


# ============================================================================
# 5. Visualisation
# ============================================================================

def plot_elbow_silhouette(results: list[dict], best_k: int) -> None:
    ks              = [r["k"]                        for r in results]
    train_inertias  = [r["train"]["inertia"]         for r in results]
    val_inertias    = [r["val"]["inertia"]            for r in results]
    train_sil       = [r["train"]["silhouette"]      for r in results]
    val_sil         = [r["val"]["silhouette"]        for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("KMeans — Model Selection", fontsize=14, fontweight="bold")

    # Elbow
    ax = axes[0]
    ax.plot(ks, train_inertias, "o-", label="Train inertia", color="steelblue")
    ax.plot(ks, val_inertias,   "s--", label="Val inertia",  color="tomato")
    ax.axvline(best_k, color="green", linestyle=":", linewidth=1.8,
               label=f"Best k={best_k}")
    ax.set_title("Elbow Curve")
    ax.set_xlabel("Number of clusters k")
    ax.set_ylabel("Inertia (WCSS)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Silhouette
    ax = axes[1]
    ax.plot(ks, train_sil, "o-",  label="Train silhouette", color="steelblue")
    ax.plot(ks, val_sil,   "s--", label="Val silhouette",   color="tomato")
    ax.axvline(best_k, color="green", linestyle=":", linewidth=1.8,
               label=f"Best k={best_k}")
    ax.set_title("Silhouette Score")
    ax.set_xlabel("Number of clusters k")
    ax.set_ylabel("Silhouette score  (higher → better)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("model_selection.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: model_selection.png")


def plot_clusters_pca(
    X_train: np.ndarray,
    X_val: np.ndarray,
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    centers: np.ndarray,
    k: int,
) -> None:
    """Project to 2-D via PCA and plot train / val clusters side by side."""
    pca = PCA(n_components=2, random_state=42)
    pca.fit(X_train)

    Xtr_2d  = pca.transform(X_train)
    Xval_2d = pca.transform(X_val)
    Cen_2d  = pca.transform(centers)

    cmap   = plt.get_cmap("tab10")
    colors = [cmap(i / k) for i in range(k)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"KMeans Clusters (k={k}) — PCA projection", fontsize=14,
                 fontweight="bold")

    for ax, X2d, labels, title in [
        (axes[0], Xtr_2d,  train_labels, "Training set"),
        (axes[1], Xval_2d, val_labels,   "Validation set"),
    ]:
        for c in range(k):
            mask = labels == c
            ax.scatter(
                X2d[mask, 0], X2d[mask, 1],
                s=20, alpha=0.6, color=colors[c], label=f"Cluster {c}"
            )
        # Plot centroids (projected)
        ax.scatter(
            Cen_2d[:, 0], Cen_2d[:, 1],
            s=200, marker="*", c="black", zorder=5, label="Centroids"
        )
        ax.set_title(title)
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("clusters_pca.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: clusters_pca.png")


def print_metrics_table(train_m: dict, val_m: dict) -> None:
    print("\n" + "=" * 52)
    print(f"{'Metric':<25} {'Train':>12} {'Val':>12}")
    print("-" * 52)
    for key in ("inertia", "silhouette", "calinski_harabasz", "davies_bouldin"):
        t = train_m.get(key, float("nan"))
        v = val_m.get(key, float("nan"))
        print(f"  {key:<23} {t:>12.4f} {v:>12.4f}")
    print("=" * 52)
    print("  silhouette      : higher is better  (range −1 … 1)")
    print("  calinski_harabasz: higher is better")
    print("  davies_bouldin  : lower  is better")
    print("=" * 52 + "\n")
