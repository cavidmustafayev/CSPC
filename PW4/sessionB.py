import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier

def agreement_score(cultivar, labels):
    ct = pd.crosstab(cultivar, labels)
    return ct.max(axis=0).sum() / len(cultivar), ct

def main():
    os.makedirs("results", exist_ok=True)
    wine = load_wine(as_frame=True)
    X = wine.data
    cultivar = wine.target
    print(wine.DESCR[:600])
    print(X.describe().T[["min", "max", "std"]])

    pca_raw = PCA().fit(X)
    print("\nB2 Raw PC1 explained variance ratio:", round(pca_raw.explained_variance_ratio_[0], 4))
    raw_loadings = pd.Series(pca_raw.components_[0], index=X.columns).sort_values(key=abs, ascending=False)
    print("B2 Raw PC1 loadings:\n", raw_loadings)

    scaler = StandardScaler()
    Z = scaler.fit_transform(X)
    pca_scaled = PCA().fit(Z)
    evr = pca_scaled.explained_variance_ratio_
    cum_evr = np.cumsum(evr)
    for i, (v, c) in enumerate(zip(evr, cum_evr), 1):
        print(f"PC{i}: {v:.4f} (cum: {c:.4f})")

    plt.figure(figsize=(7, 4))
    pcs = range(1, len(evr) + 1)
    plt.bar(pcs, evr, alpha=0.7, label="Individual")
    plt.plot(pcs, cum_evr, marker="o", color="black", label="Cumulative")
    plt.axhline(0.9, color="red", linestyle="--", label="90% threshold")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/B3_scree.png")
    plt.close()

    loadings_df = pd.DataFrame({
        "PC1": pca_scaled.components_[0],
        "PC2": pca_scaled.components_[1]
    }, index=X.columns).sort_values("PC1", ascending=False)
    print("\nB3 Scaled PC1 & PC2 loadings:\n", loadings_df)

    Z_pca = pca_scaled.transform(Z)
    plt.figure(figsize=(6, 4))
    plt.scatter(Z_pca[:, 0], Z_pca[:, 1], alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig("results/B3_pca_unlabeled.png")
    plt.close()

    inertias_rs = [KMeans(n_clusters=3, n_init=1, random_state=rs).fit(Z).inertia_ for rs in range(8)]
    print("\nB4a Inertias (n_init=1, rs=0..7):", np.round(inertias_rs, 2))

    ks = range(1, 11)
    inertias_k = [KMeans(n_clusters=k, n_init=10, random_state=0).fit(Z).inertia_ for k in ks]
    plt.figure(figsize=(6, 4))
    plt.plot(ks, inertias_k, marker="o")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.savefig("results/B4_elbow.png")
    plt.close()

    km_scaled = KMeans(n_clusters=3, n_init=10, random_state=0).fit(Z)
    print("\nB5 Cluster sizes:", np.bincount(km_scaled.labels_))
    centers_real = pd.DataFrame(scaler.inverse_transform(km_scaled.cluster_centers_), columns=X.columns)
    print("B5 Cluster centers (real units):\n", centers_real.round(2).to_string())

    plt.figure(figsize=(6, 4))
    plt.scatter(Z_pca[:, 0], Z_pca[:, 1], c=km_scaled.labels_, cmap="viridis", alpha=0.8)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig("results/B5_pca_clusters.png")
    plt.close()

    km_raw = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
    km_pc2 = KMeans(n_clusters=3, n_init=10, random_state=0).fit(Z_pca[:, :2])

    agr_scaled, ct_scaled = agreement_score(cultivar, km_scaled.labels_)
    agr_raw, ct_raw = agreement_score(cultivar, km_raw.labels_)
    agr_pc2, ct_pc2 = agreement_score(cultivar, km_pc2.labels_)

    print("\nCrosstab (Scaled):\n", ct_scaled)
    print("\nCrosstab (Raw):\n", ct_raw)
    print("\nCrosstab (2 PCs):\n", ct_pc2)

    agr_df = pd.DataFrame([
        {"representation": "Scaled (13 features)", "agreement": round(agr_scaled, 4)},
        {"representation": "Raw unscaled (13 features)", "agreement": round(agr_raw, 4)},
        {"representation": "Scaled PCA (first 2 PCs)", "agreement": round(agr_pc2, 4)}
    ])
    agr_df.to_csv("results/B_agreement.csv", index=False)
    print("\nB6 Agreement Table:\n", agr_df.to_string(index=False))

    X_tr, X_te, y_tr, y_te = train_test_split(X, cultivar, test_size=0.25, random_state=42, stratify=cultivar)
    sup_full = make_pipeline(StandardScaler(), KNeighborsClassifier(5)).fit(X_tr, y_tr)
    sup_pca2 = make_pipeline(StandardScaler(), PCA(n_components=2), KNeighborsClassifier(5)).fit(X_tr, y_tr)
    print(f"\nBonus Supervised KNN -> Full: {sup_full.score(X_te, y_te):.4f}, PCA(2): {sup_pca2.score(X_te, y_te):.4f}")

if __name__ == "__main__":
    main()