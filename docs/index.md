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

### 2. Data Validation

Ensure your simulation outputs are statistically valid before processing.

- **Type Checking:** Verifies that input dataframes contain strictly numeric values for signals and proper string identifiers for variables.
- **Sanity Checks:** Detects overlap between input and outcome variables and flags insufficient sample sizes (\<10 rows) to prevent unstable analysis.

### 3. Reliability Analysis (In Development)

Implementing the generalised framework by Malkiel et al. (2025).

-   **Relaxed Assumptions:** Moves beyond the rigid constraints of the classical $\hat{a}$-versus-$a$ method.
-   **Advanced Regression:** Fits nonlinear expectation models and models non-constant variance (heteroscedasticity) using Kernel Average Smoothers.
-   **Uncertainty Quantification:** Uses bootstrap resampling to generate robust confidence bounds for Probability of Detection (PoD) curves.

------------------------------------------------------------------------

## Installation

[![CI Status](https://github.com/JGIBristol/digiqual/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/JGIBristol/digiqual/actions) [![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/) [![License](https://img.shields.io/github/license/JGIBristol/digiqual)](LICENSE)

You can install `digiqual` directly from GitHub.

### Option 1: Install via uv (Recommended)

If you are managing a project with `uv`, add `digiqual` as a dependency:

- To install the latest stable release (v0.4.1):
```bash
uv add "digiqual @ git+https://github.com/JGIBristol/digiqual.git@v0.4.1"
```

- To install the latest development version (main branch):
```bash
uv add "digiqual @ git+https://github.com/JGIBristol/digiqual.git"
```

If you just want to install it into a virtual environment without modifying a project file (e.g., for a quick script), use pip interface:

```bash
uv pip install "git+https://github.com/JGIBristol/digiqual.git@v0.4.1"
```

### Option 2: Install via standard pip

To install the latest stable release (v0.4.1):

```bash
pip install "git+https://github.com/JGIBristol/digiqual.git@v0.4.1"
```
To install the latest development version:

```bash
pip install "git+https://github.com/JGIBristol/digiqual.git"
```
------------------------------------------------------------------------

## References

This package implements methods described in:

**Malkiel, N., Croxford, A. J., & Wilcox, P. D. (2025).** A generalized method for the reliability assessment of safety–critical inspection. Proceedings of the Royal Society A, 481: 20240654. https://doi.org/10.1098/rspa.2024.0654
