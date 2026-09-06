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