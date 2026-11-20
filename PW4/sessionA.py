import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.inspection import permutation_importance

def main():
    os.makedirs("results", exist_ok=True)
    df_raw = pd.read_csv("data/concrete.csv")
    print("Duplicates:", df_raw.duplicated().sum())
    print("Missing:\n", df_raw.isna().sum())
    df = df_raw.drop_duplicates().reset_index(drop=True)
    print(df.describe())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(df["cement"], df["strength"], alpha=0.4)
    axes[0].set_xlabel("Cement (kg/m3)")
    axes[0].set_ylabel("Strength (MPa)")
    axes[1].scatter(df["age"], df["strength"], alpha=0.4)
    axes[1].set_xlabel("Age (days)")
    axes[1].set_ylabel("Strength (MPa)")
    plt.tight_layout()
    plt.savefig("results/A2_scatter.png")
    plt.close()

    X = df.drop(columns=["strength"])
    y = df["strength"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Train size:", len(X_train), "Test size:", len(X_test))

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_train_pred = lin_reg.predict(X_train)
    y_test_pred = lin_reg.predict(X_test)

    n_train, p = X_train.shape
    r2_tr = r2_score(y_train, y_train_pred)
    adj_r2_tr = 1 - (1 - r2_tr) * (n_train - 1) / (n_train - p - 1)
    r2_te = r2_score(y_test, y_test_pred)
    rmse_tr = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_te = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae_te = mean_absolute_error(y_test, y_test_pred)

    print("Intercept:", lin_reg.intercept_)
    print("Coefficients:", dict(zip(X.columns, lin_reg.coef_)))
    print(f"A4 -> Train RMSE: {rmse_tr:.2f}, Test RMSE: {rmse_te:.2f}, Test MAE: {mae_te:.2f}, Train R2: {r2_tr:.4f}, Adj R2: {adj_r2_tr:.4f}, Test R2: {r2_te:.4f}")

    residuals = y_test - y_test_pred
    plt.figure(figsize=(6, 4))
    plt.scatter(X_test["age"], residuals, alpha=0.5)
    plt.axhline(0, color="red", linestyle="--")
    plt.xscale("log")
    plt.xlabel("Age (days, log scale)")
    plt.ylabel("Residuals (MPa)")
    plt.tight_layout()
    plt.savefig("results/A4_residuals.png")
    plt.close()

    poly_rows = []
    for d in [1, 2, 3, 4]:
        poly = make_pipeline(PolynomialFeatures(degree=d, include_bias=False), StandardScaler(), LinearRegression())
        poly.fit(X_train, y_train)
        p_d = poly[0].n_output_features_
        tr_p = poly.predict(X_train)
        te_p = poly.predict(X_test)
        r2_d_tr = r2_score(y_train, tr_p)
        adj_d_tr = 1 - (1 - r2_d_tr) * (n_train - 1) / (n_train - p_d - 1) if n_train > p_d + 1 else np.nan
        poly_rows.append({
            "degree": d,
            "p": p_d,
            "train_rmse": np.sqrt(mean_squared_error(y_train, tr_p)),
            "test_rmse": np.sqrt(mean_squared_error(y_test, te_p)),
            "train_r2": r2_d_tr,
            "adj_r2": adj_d_tr,
            "test_r2": r2_score(y_test, te_p)
        })
    print("\nA5 Polynomial Table:\n", pd.DataFrame(poly_rows).to_string(index=False))

    y_class = (df["strength"] >= 40).astype(int)
    Xc_train, Xc_test, c_train, c_test = train_test_split(X, y_class, test_size=0.2, random_state=42, stratify=y_class)

    ks = [1, 3, 5, 7, 15, 25, 51, 101]
    knn_tr_acc, knn_te_acc = [], []
    for k in ks:
        knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
        knn.fit(Xc_train, c_train)
        knn_tr_acc.append(knn.score(Xc_train, c_train))
        knn_te_acc.append(knn.score(Xc_test, c_test))

    plt.figure(figsize=(6, 4))
    plt.plot(ks, knn_tr_acc, marker="o", label="Train")
    plt.plot(ks, knn_te_acc, marker="s", label="Test")
    plt.xscale("log")
    plt.xlabel("K (log scale)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/A6_knn.png")
    plt.close()

    best_k = 5
    knn_scaled = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=best_k)).fit(Xc_train, c_train)
    knn_unscaled = KNeighborsClassifier(n_neighbors=best_k).fit(Xc_train, c_train)
    print(f"\nKNN (K={best_k}) Scaled Test Acc: {knn_scaled.score(Xc_test, c_test):.4f}, Unscaled: {knn_unscaled.score(Xc_test, c_test):.4f}")

    split_accs = []
    for rs in range(5):
        xtr, xte, ytr, yte = train_test_split(X, y_class, test_size=0.2, random_state=rs, stratify=y_class)
        m = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=best_k)).fit(xtr, ytr)
        split_accs.append(m.score(xte, yte))
    print("KNN 5-split accuracies:", np.round(split_accs, 4), "Spread:", round(max(split_accs) - min(split_accs), 4))

    depths = [1, 2, 3, 4, 5, 6, 8, None]
    tree_rows = []
    for d in depths:
        dt = DecisionTreeClassifier(criterion="entropy", max_depth=d, min_samples_leaf=1, random_state=0)
        dt.fit(Xc_train, c_train)
        tree_rows.append({
            "max_depth": str(d),
            "leaves": dt.get_n_leaves(),
            "train_acc": dt.score(Xc_train, c_train),
            "test_acc": dt.score(Xc_test, c_test)
        })
    tree_df = pd.DataFrame(tree_rows)
    print("\nA6 Tree Table:\n", tree_df.to_string(index=False))

    plt.figure(figsize=(6, 4))
    plt.plot(tree_df["max_depth"], tree_df["train_acc"], marker="o", label="Train")
    plt.plot(tree_df["max_depth"], tree_df["test_acc"], marker="s", label="Test")
    plt.xlabel("max_depth")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/A6_tree.png")
    plt.close()

    best_tree = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=0).fit(Xc_train, c_train)
    print("\nTop 2 levels of tree:\n", export_text(best_tree, feature_names=list(X.columns), max_depth=2))

    metrics_rows = []
    for name, clf in [(f"KNN (K={best_k})", knn_scaled), ("Decision Tree (depth=5)", best_tree)]:
        preds = clf.predict(Xc_test)
        tn, fp, fn, tp = confusion_matrix(c_test, preds).ravel()
        metrics_rows.append({
            "model": name,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "accuracy": accuracy_score(c_test, preds),
            "precision": precision_score(c_test, preds),
            "recall": recall_score(c_test, preds),
            "f1": f1_score(c_test, preds)
        })
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv("results/A_metrics.csv", index=False)
    print("\nA7 Final Metrics:\n", metrics_df.to_string(index=False))

    probas = best_tree.predict_proba(Xc_test)[:, 1]
    print("\nA7 Threshold Sweep (Decision Tree):")
    for t in [0.3, 0.5, 0.7, 0.9]:
        p_t = (probas >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(c_test, p_t).ravel()
        print(f"t={t}: TN={tn}, FP={fp}, FN={fn}, TP={tp} | Precision={precision_score(c_test, p_t):.4f}, Recall={recall_score(c_test, p_t):.4f}")

    std_reg = make_pipeline(StandardScaler(), LinearRegression()).fit(X_train, y_train)
    perm_knn = permutation_importance(knn_scaled, Xc_test, c_test, scoring="f1", n_repeats=20, random_state=0)
    perm_tree = permutation_importance(best_tree, Xc_test, c_test, scoring="f1", n_repeats=20, random_state=0)

    imp_df = pd.DataFrame({
        "feature": X.columns,
        "raw_coef": lin_reg.coef_,
        "std_coef": std_reg[-1].coef_,
        "tree_imp": best_tree.feature_importances_,
        "perm_knn": perm_knn.importances_mean,
        "perm_tree": perm_tree.importances_mean
    })
    print("\nA8 Feature Importances:\n", imp_df.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].barh(imp_df["feature"], np.abs(imp_df["std_coef"]))
    axes[0].set_title("|Standardized Coef|")
    axes[1].barh(imp_df["feature"], imp_df["tree_imp"])
    axes[1].set_title("Tree Importance")
    axes[2].barh(imp_df["feature"], imp_df["perm_knn"])
    axes[2].set_title("KNN Permutation (F1)")
    plt.tight_layout()
    plt.savefig("results/A8_importance.png")
    plt.close()

    X_b1 = X.copy()
    X_b1["age"] = np.log(X_b1["age"])
    xtr1, xte1, ytr1, yte1 = train_test_split(X_b1, y, test_size=0.2, random_state=42)
    m1 = LinearRegression().fit(xtr1, ytr1)

    X_b2 = X_b1.drop(columns=["water"]).copy()
    X_b2["w_b"] = X["water"] / (X["cement"] + X["slag"] + X["ash"])
    xtr2, xte2, ytr2, yte2 = train_test_split(X_b2, y, test_size=0.2, random_state=42)
    m2 = LinearRegression().fit(xtr2, ytr2)
    print(f"\nBonus Physics R2 -> log(age): {r2_score(yte1, m1.predict(xte1)):.4f}, log(age) + w/b: {r2_score(yte2, m2.predict(xte2)):.4f}")

if __name__ == "__main__":
    main()