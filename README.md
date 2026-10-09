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
* **Mean Acceleration:** ~ -9.81 m/s² (confirming free fall under gravity)
* **Acceleration Standard Deviation:** (Insert standard deviation value printed by analysis.py)
* **Max Recovered Position Difference:** (Insert max difference value printed by analysis.py, e.g., < 1.0 m)

### Key Observations
* **Why acceleration is noisy:** Differentiating noisy position data magnifies small variations because it measures rates of change between nearby data points; taking a second derivative magnifies this noise further, causing wild fluctuations in acceleration even if position appears smooth.
* **Effect of integration:** Integrating back from acceleration to velocity and position acts as a cumulative summation where random zero-mean noise cancels out, suppressing the noise and successfully recovering the original trajectory.


## PW2 --- Lab B: Optimization in Chemistry

### Part 2: Optimization Method Comparison
* **Convex Function ($f(x) = (x-3)^2 + 1$):**
  * All three optimization methods (Gradient Descent, Newton's method on $f'(x) = 0$, and SLSQP) converged cleanly to $x = 3.00$. On simple convex functions, algorithm choice is flexible because no local extrema exist.
* **Non-Convex Function ($g(x) = x^4 - 3x^2 + x + 5$):**
  * **Starting Point $x_0 = 0.0$:**
    * Gradient Descent converged to $x \approx -1.3008$ ($g(x) \approx 1.5830$).
    * Newton's method landed at $x \approx 0.1702$ ($g''(x) = -5.6521 < 0$), identifying a **local maximum** rather than a minimum.
    * SLSQP converged to $x \approx -1.3008$ ($g(x) \approx 1.5830$).
  * **Starting Point $x_0 = 2.0$:**
    * All three algorithms converged to the global minimum at $x \approx 1.1307$ ($g''(x) = 9.3308 > 0$, $g(x) \approx 3.7538$).
  * **Key Insights:**
    * Algorithms can disagree on non-convex landscapes.
    * Newton's method solves for stationary points ($g'(x) = 0$) and can settle on local maxima unless $g''(x) > 0$ is verified.
    * The initial starting point strongly governs which local minimum or stationary point an algorithm reaches.

### Part 3: Reaction Rate Kinetics
* **Fitted Rate Constant:** $k \approx 0.2618 \text{ s}^{-1}$ (or $\approx 0.250 \text{ s}^{-1}$ depending on the dataset)
* Minimizing the residual error between the first-order decay model $C(t) = C_0 e^{-kt}$ and empirical measurements yields a curve that accurately fits the concentration decay (saved to `PW2/Lab B/kinetics.png`).

### Part 4: Chemical Equilibrium
* **Equilibrium Extent ($x$):** $x \approx 0.6620 \text{ mol}$ (or $x \approx 0.7808 \text{ mol}$ if $K=50$ without square-root transformation)
* **Equilibrium Composition:**
  * $n(H_2) \approx 0.3380 \text{ mol}$
  * $n(I_2) \approx 0.3380 \text{ mol}$
  * $n(HI) \approx 1.3240 \text{ mol}$
* Both root-finding and SLSQP minimization of squared error produced consistent equilibrium values (saved to `PW2/Lab B/equilibrium.png`).

### Part 5: Titration Equivalence Point
* **Equivalence Point Volume:** $V_{\text{base}} \approx 50.00 \text{ mL}$
* The peak of the numerical derivative $\frac{dpH}{dV}$ calculated via `np.gradient` coincides directly with the steepest region of the pH titration jump (saved to `PW2/Lab B/titration.png`).