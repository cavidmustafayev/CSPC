import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats 
import numpy as np
df_heart = pd.read_csv("Heart_disease_cleveland_new.csv")
print("Heart Dataset Shape:", df_heart.shape)
print(df_heart.head())
cols = ["age", "chol", "trestbps", "thalach"]
df_continuous = df_heart[cols]
print(df_continuous.describe())
fig, axes=plt.subplots(2, 2, figsize=(12, 10))
for ax, col in zip(axes.flatten(), cols):
    ax.hist(df_continuous[col], bins=20, edgecolor="black", alpha=0.7)
    ax.set_title(f"Histogram of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
plt.tight_layout()
plt.savefig("heart_histograms.png")
plt.close()
for col in cols:
    stat, p_value=stats.shapiro(df_heart[col])
    print(f"{col}-Shapiro-Wilk Test: Statistics={stat:.3f}, p-value={p_value:.3f}")
thalach_disease=df_heart[df_heart["target"]==1]["thalach"]
thalach_healthy=df_heart[df_heart["target"]==0]["thalach"]
t_stat, p_val = stats.ttest_ind(thalach_disease, thalach_healthy)
print(f"t-statistic: {t_stat:.4f}, p-value: {p_val:.4e}")
mean_dis, sem_dis = np.mean(thalach_disease), stats.sem(thalach_disease)
mean_hea, sem_hea = np.mean(thalach_healthy), stats.sem(thalach_healthy)

plt.figure(figsize=(8,6))
plt.bar(["Disease", "Healthy"], [mean_dis, mean_hea], yerr=[sem_dis, sem_hea], capsize=5, color=["red", "blue"])
plt.ylabel("Mean Max HR")
plt.title("Mean thalach")
plt.savefig("thalach_comparison.png")
plt.close()

r_val, p_val_corr=stats.pearsonr(df_heart["age"], df_heart["thalach"])
print(f"Pearson correlation between age and tcalach: r={r_val:.4f}, p value={p_val_corr:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(df_heart["age"], df_heart["thalach"], alpha=0.7, edgecolors="k")
plt.xlabel("Age")
plt.ylabel("Thalach")
plt.title("Age vs Thalach")
z=np.polyfit(df_heart["age"], df_heart["thalach"], 1)
p=np.poly1d(z)
plt.plot(df_heart["age"], p(df_heart["age"]), color="red", linestyle="--", linewidth=2)

plt.savefig("age_thalach_scatter.png")
plt.close()

df_chem = pd.read_csv("chemicals_cancer.csv")
r_benz_naive, _ = stats.pearsonr(df_chem["benzene"], df_chem["malignancy"])
r_cadm_naive, _ = stats.pearsonr(df_chem["cadmium"], df_chem["malignancy"])
print(f"\nNaive Correlation with Malignancy:")
print(f"  Benzene: {r_benz_naive:.4f}")
print(f"  Cadmium: {r_cadm_naive:.4f}")
p_med = df_chem["pollution_index"].median()
df_stratified = df_chem[(df_chem["pollution_index"] > p_med - 10) & (df_chem["pollution_index"] < p_med + 10)]
r_benz_strat, _ = stats.pearsonr(df_stratified["benzene"], df_stratified["malignancy"])
r_cadm_strat, _ = stats.pearsonr(df_stratified["cadmium"], df_stratified["malignancy"])
print(f"\nStratified Correlation (pollution_index ~ {p_med:.1f}):")
print(f"  Benzene: {r_benz_strat:.4f}")
print(f"  Cadmium: {r_cadm_strat:.4f}")
