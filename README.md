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
- **Analytical Law Fit:** The experimental scatter plot matches the theoretical curve $N_0 e^{-\lambda t}$ ($\lambda = 0.3$) very closely. Side-by-side comparison with shared axes confirms that the empirical data aligns directly with the analytical decay law[cite: 1].

### Snakemake Pipeline
- **Automation:** A `Snakefile` was implemented to manage figure creation via `plot.py`[cite: 1].
- **Dependency Tracking:** Snakemake checks file timestamps so `figure.png` is only generated when `decay_observed.csv` or `plot.py` changes, preventing redundant computations[cite: 1].