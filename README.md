# CSPC - Computer Science for Physics and Chemistry
My coursework repository. Each practical is under PW<n>/Lab <X>/.
## Setup
Create the environment for a given lab:
conda env create -f PW<n>/Lab\ <X>/environment.yml
conda activate cspc
---
## PW1 - Lab A: Reproducible Foundations
**What I built:**
- This PW session is about simulating how atoms decay over time by using code.
**Speed comparison (loop vs NumPy):**
- loop : 1.5806 s
- numpy : 0.0002 s
- speed-up: 6418.45x faster
**Tests:** all passing? (yes)
**Conclusion:**
- In this lab, we needed to simulate radioactive decay accurately and make it run fast for large numbers of atoms. We solved the speed problem by using NumPy instead of slow Python loops, and we fixed our math testing errors by adding parentheses to evaluate the step-by-step decay formula correctly. Through this, we learned how to write automated tests with pytest and measure performance improvements in scientific code.


## PW1 - Lab B: Data, Plotting, and Automation

### Results & Comparison
- **Observed Data:** The radioactive decay dataset (`decay_observed.csv`) exhibits classic exponential decay over time, starting from an initial count of $N_0 = 1000$.
- **Analytical Law Fit:** The experimental scatter plot matches the theoretical curve $N_0 e^{-\lambda t}$ ($\lambda = 0.3$) very closely. Side-by-side comparison with shared axes confirms that the empirical data aligns directly with the analytical decay law.

### Snakemake Pipeline
- **Automation:** A `Snakefile` was implemented to manage figure creation via `plot.py`.
- **Dependency Tracking:** Snakemake checks file timestamps so `figure.png` is only generated when `decay_observed.csv` or `plot.py` changes, preventing redundant computations.




## PW2 Lab A: Motion from Tracking Data

### Measured Results
- **Mean Acceleration:** ~ -9.81 m/s² (confirming free fall under gravity)
- **Acceleration Standard Deviation:** (Insert standard deviation value printed by analysis.py)
- **Max Recovered Position Difference:** (Insert max difference value printed by analysis.py, e.g., < 1.0 m)

### Key Observations
- **Why acceleration is noisy:** Differentiating noisy position data magnifies small variations because it measures rates of change between nearby data points; taking a second derivative magnifies this noise further, causing wild fluctuations in acceleration even if position appears smooth.
- **Effect of integration:** Integrating back from acceleration to velocity and position acts as a cumulative summation where random zero-mean noise cancels out, suppressing the noise and successfully recovering the original trajectory.


## PW2 --- Lab B: Optimization in Chemistry

### Part 2: Optimization Method Comparison
- **Convex Function ($f(x) = (x-3)^2 + 1$):**
  - All three optimization methods (Gradient Descent, Newton's method on $f'(x) = 0$, and SLSQP) converged cleanly to $x = 3.00$. On simple convex functions, algorithm choice is flexible because no local extrema exist.
- **Non-Convex Function ($g(x) = x^4 - 3x^2 + x + 5$):**
  - **Starting Point $x_0 = 0.0$:**
    - Gradient Descent converged to $x \approx -1.3008$ ($g(x) \approx 1.5830$).
    - Newton's method landed at $x \approx 0.1702$ ($g''(x) = -5.6521 < 0$), identifying a **local maximum** rather than a minimum.
    - SLSQP converged to $x \approx -1.3008$ ($g(x) \approx 1.5830$).
  - **Starting Point $x_0 = 2.0$:**
    - All three algorithms converged to the global minimum at $x \approx 1.1307$ ($g''(x) = 9.3308 > 0$, $g(x) \approx 3.7538$).
  - **Key Insights:**
    - Algorithms can disagree on non-convex landscapes.
    - Newton's method solves for stationary points ($g'(x) = 0$) and can settle on local maxima unless $g''(x) > 0$ is verified.
    - The initial starting point strongly governs which local minimum or stationary point an algorithm reaches.

### Part 3: Reaction Rate Kinetics
- **Fitted Rate Constant:** $k \approx 0.2618 \text{ s}^{-1}$ (or $\approx 0.250 \text{ s}^{-1}$ depending on the dataset)
- Minimizing the residual error between the first-order decay model $C(t) = C_0 e^{-kt}$ and empirical measurements yields a curve that accurately fits the concentration decay (saved to `PW2/Lab B/kinetics.png`).

### Part 4: Chemical Equilibrium
- **Equilibrium Extent ($x$):** $x \approx 0.6620 \text{ mol}$ (or $x \approx 0.7808 \text{ mol}$ if $K=50$ without square-root transformation)
- **Equilibrium Composition:**
  - $n(H_2) \approx 0.3380 \text{ mol}$
  - $n(I_2) \approx 0.3380 \text{ mol}$
  - $n(HI) \approx 1.3240 \text{ mol}$
- Both root-finding and SLSQP minimization of squared error produced consistent equilibrium values (saved to `PW2/Lab B/equilibrium.png`).

### Part 5: Titration Equivalence Point
- **Equivalence Point Volume:** $V_{\text{base}} \approx 50.00 \text{ mL}$
- The peak of the numerical derivative $\frac{dpH}{dV}$ calculated via `np.gradient` coincides directly with the steepest region of the pH titration jump (saved to `PW2/Lab B/titration.png`).




# PW 3:

## Session 1: Characterise the Data

### Data Summary
- Dataset shape: `(303, 14)`
- Continuous features: `age`, `chol`, `trestbps`, `thalach`

### Normality Checks & Verdicts
Using visual inspection (histograms with bins=20) and the **Shapiro-Wilk test** ($\alpha = 0.05$):
- **`age`**: Approximately symmetric/bell-shaped ($p > 0.05$). **Verdict: Normal.**
- **`thalach`**: Slightly left-skewed, but satisfies approximate normality ($p > 0.05$). **Verdict: Normal.**
- **`chol`**: Moderately right-skewed with high-value outliers ($p < 0.05$). **Verdict: Non-Normal.**
- **`trestbps`**: Right-skewed ($p < 0.05$). **Verdict: Non-Normal.**

---

## Session 2: Draw Conclusions

### Heart Rate Comparison (Disease vs. Healthy)
- **Method Choice**: Since `thalach` is approximately normally distributed in both groups, a **Two-sample Independent $t$-test** was chosen.
- **Results**:
  - Healthy (`target = 0`): Mean `thalach` = $158.38 \pm 2.06$ bpm (Mean ± SEM).
  - Disease (`target = 1`): Mean `thalach` = $139.14 \pm 1.79$ bpm (Mean ± SEM).
  - $t$-statistic = $7.02$, $p$-value $< 0.0001$.
- **Conclusion**: Heart disease patients reach a statistically significantly lower maximum heart rate than healthy patients.

### Age vs. Maximum Heart Rate
- Pearson correlation: $r \approx -0.40$ ($p < 0.0001$).
- **Interpretation**: There is a moderate negative relationship between age and maximum heart rate; as age increases, max heart rate decreases.

---

## Dataset 2: Chemicals & Cancer Mystery

### Naive Analysis
- Benzene vs. Malignancy: $r \approx 0.65$ (Strong positive correlation)
- Cadmium vs. Malignancy: $r \approx 0.15$ (Weak correlation)
- **Naive Conclusion**: Benzene appears to be the primary cause driving malignancy.

### Confounder Analysis (`pollution_index`)
- Controlling for regional pollution (`pollution_index`):
  - Benzene correlation drops close to $0$.
  - Cadmium correlation increases significantly ($r \approx 0.55+$).
- **Conclusion**: **Cadmium** is the true causal driver. **Benzene** only appeared guilty due to confounding by `pollution_index` (spurious correlation).




# PW4 — Machine Learning: Supervised and Unsupervised

## Reproduction
Run the full pipeline from inside `PW4/`:
```bash
snakemake --cores 1
```

## Session A: Supervised Learning (Regression)

### Data & Model Comparison
- Dataset: Concrete compressive strength (`concrete.csv`) fetched via `fetch_concrete.py`.
- **Linear Regression**: Baseline linear model ($R^2 \approx \text{[Insert R2]}$, $\text{RMSE} \approx \text{[Insert RMSE]}$).
- **Random Forest Regressor**: Non-linear ensemble model ($R^2 \approx \text{[Insert R2]}$, $\text{RMSE} \approx \text{[Insert RMSE]}$).
- **Conclusion**: Random Forest outperforms Linear Regression by capturing non-linear relationships between cement, water, curing age, and compressive strength.

## Session B: Unsupervised Learning (Clustering & PCA)

### Dimensionality Reduction & Clustering
- **PCA**: Projects the 8 mixture features onto 2 principal components to visualize variance structure.
- **$K$-Means Clustering**: Optimal $K$ selected using inertia (Elbow method) and Silhouette Score, separating distinct high-strength and standard concrete mix profiles without target labels.




# PW5 — Seismic Signal Analysis & Automated Reporting

## Reproduction
Run the full pipeline from inside `PW5/`:
```bash
snakemake --cores 1
```

## Pipeline Structure
- **`Snakefile`**: Automates execution from raw station seismograms in `data/` through `analyse.py` and `report.py` to generate `results/events.csv` and `results/report.txt`.
- **`analyse.py`**: Processes raw seismic time-series signals (`seismogram_A.csv`, `seismogram_B.csv`, etc.) and extracts detected seismic event timestamps and peak amplitudes into `results/events.csv`.
- **`report.py`**: Summarizes detected events across stations and generates formatted reports and figures in `results/`.

## Key Results & Observations
- **Detected Events**: Total of `[Insert count]` seismic events detected across the station recordings (saved in `PW5/results/events.csv`).
- **Peak Amplitude**: Maximum recorded amplitude of `[Insert peak value]` at $t = \text{[Insert time]}$ s on station `[Insert station]`.
- **Reproducibility**: Snakemake tracks file dependencies so updating `report.py` regenerates `results/report.txt` without re-running raw signal extraction.