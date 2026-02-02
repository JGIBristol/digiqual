# digiqual

 **Statistical Toolkit for Reliability Assessment in NDT**

`digiqual` is a Python library designed for Non-Destructive Evaluation (NDE) engineers. It provides a robust statistical framework for performing Model-Assisted Probability of Detection (MAPOD) studies and reliability assessments.

The package is built to implement the Generalised $\hat{a}$-versus-$a$ Method, allowing users to assess inspection reliability even when traditional assumptions (linearity, constant variance, Gaussian noise) are not met.

------------------------------------------------------------------------

## Core Features

### 1. Experimental Design

Before running expensive Finite Element (FE) simulations, `digiqual` helps you design your experiment efficiently.

- **Latin Hypercube Sampling (LHS):** Generate space-filling experimental designs to cover your deterministic parameter space (e.g., defect size) and stochastic nuisance parameters (e.g., roughness, orientation).
- **Scale & Bound:** Automatically scale samples to your specific variable bounds.

### 2. Data Validation & Diagnostics

Ensure your simulation outputs are statistically valid before processing.

- **Sanity Checks:** Detects overlap between variables, type errors, and insufficient sample sizes.
- **Sufficiency Diagnostics:** rigorous statistical tests to flag issues like "Input Coverage Gaps" or "Model Instability" before you trust the results.

### 3. Adaptive Refinement (Active Learning)

`digiqual` closes the loop between analysis and design. Instead of guessing where to run more simulations, use the `refine()` method to:

- **Fill Gaps:** Automatically detect and target empty regions in your input space.
- **Reduce Uncertainty:** Use bootstrap committees to find regions of high model variance and suggest new points exactly where the model is "confused."

### 4. Reliability Analysis (In Development)

Implementing the generalised framework by Malkiel et al. (2025).

-   **Relaxed Assumptions:** Moves beyond the rigid constraints of the classical $\hat{a}$-versus-$a$ method.
-   **Uncertainty Quantification:** Uses bootstrap resampling to generate robust confidence bounds for Probability of Detection (PoD) curves.
------------------------------------------------------------------------

## Installation

[![CI Status](https://github.com/JGIBristol/digiqual/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/JGIBristol/digiqual/actions) [![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/) [![License](https://img.shields.io/github/license/JGIBristol/digiqual)](LICENSE)

You can install `digiqual` directly from GitHub.

### Option 1: Install via uv (Recommended)

If you are managing a project with `uv`, add `digiqual` as a dependency:

- To install the latest stable release (v0.6.1):
```bash
uv add "digiqual @ git+https://github.com/JGIBristol/digiqual.git@v0.6.1"
```

- To install the latest development version (main branch):
```bash
uv add "digiqual @ git+https://github.com/JGIBristol/digiqual.git"
```

If you just want to install it into a virtual environment without modifying a project file (e.g., for a quick script), use pip interface:

```bash
uv pip install "git+https://github.com/JGIBristol/digiqual.git@v0.6.1"
```

### Option 2: Install via standard pip

To install the latest stable release (v0.6.1):

```bash
pip install "git+https://github.com/JGIBristol/digiqual.git@v0.6.1"
```
To install the latest development version:

```bash
pip install "git+https://github.com/JGIBristol/digiqual.git"
```
------------------------------------------------------------------------

## References

This package implements methods described in:

**Malkiel, N., Croxford, A. J., & Wilcox, P. D. (2025).** A generalized method for the reliability assessment of safety–critical inspection. Proceedings of the Royal Society A, 481: 20240654. https://doi.org/10.1098/rspa.2024.0654
