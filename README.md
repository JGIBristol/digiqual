# digiqual

**Statistical Toolkit for Reliability Assessment in NDT**

`digiqual` is a Python library designed for Non-Destructive Evaluation (NDE) engineers. It implements the **Generalised** $\hat{a}$-versus-a Method, allowing users to perform reliability assessments without the rigid assumptions of linearity or constant variance found in standard methods.

> **Documentation:** [Read the full documentation here](https://jgibristol.github.io/digiqual/)


## Installation

> **Just want the app?** Windows and macOS installers are attached to the [latest GitHub Release](https://github.com/JGIBristol/digiqual/releases/latest); no Python needed. See [Launch the App](https://jgibristol.github.io/digiqual/docs/gui.html) for install notes.

You can install `digiqual` directly from PyPI.

### Option 1: Install via uv (Recommended)

If you are managing a project with `uv`, add `digiqual` as a dependency:
```bash
# To install the latest stable release (v0.27.1):

uv add digiqual

# To install the latest development version (main branch from github):

uv add "digiqual @ git+https://github.com/JGIBristol/digiqual.git"
```

If you just want to install it into a virtual environment without modifying a project file (e.g., for a quick script), use pip interface:

```bash
uv pip install digiqual
```

### Option 2: Install via standard pip

To install the latest stable release (v0.27.1):

```bash
pip install digiqual
```
To install the latest development version from github:

```bash
pip install "git+https://github.com/JGIBristol/digiqual.git"
```

## Features

### 1. Experimental Design

Before running expensive Finite Element (FE) simulations, `digiqual` helps you design your experiment efficiently.

- **Latin Hypercube Sampling (LHS):** Generate space-filling designs over your parameter ranges, covering both the parameters of interest (e.g. defect size) and nuisance parameters (e.g. roughness, orientation).
- **Run in any order, stop early:** Points are reordered with a greedy max-min algorithm, so any first *k* runs are already well spread.

### 2. Data Validation & Diagnostics

Ensure your simulation outputs are statistically valid before processing.

- **Sanity Checks:** Detects overlap between variables, type errors, and insufficient sample sizes.
- **Sufficiency Diagnostics:** Five tests (input coverage gaps, model fit, two bootstrap-stability checks and collinearity/VIF) flag problems before you trust the results.

### 3. Adaptive Refinement (Active Learning)

`digiqual` closes the loop between analysis and design.

- Smart Refinement: Use `refine()` to identify specific weaknesses in your data. It uses bootstrap committees to find regions of high uncertainty and suggests new points exactly where the model is "confused".

- Automated Workflows: Use the `optimise()` method to run a fully automated "Active Learning" loop. It generates an initial design, executes your external solver, checks diagnostics, and iteratively refines the model until statistical requirements are met.

### 4. Generalised Reliability Analysis

The package includes a full statistical engine for calculating Probability of Detection (PoD) curves.

-   **Relaxed Assumptions:** Moves beyond the rigid constraints of the classical $\hat{a}$-versus-$a$ method by handling non-linear signal responses and heteroscedastic noise.
-   **Multi-Dimensional Active Marginalisation:** Resolves multidimensional physics by integrating out stochastic nuisance parameters (like roughness) via Monte Carlo methods, outputting high-fidelity 2D PoD surface heatmaps alongside standard 1D curves.
-   **Model Selection:** Compares polynomials (degree 1–10) and a Kriging (Gaussian Process) model by 10-fold cross-validation, and picks the simplest model within one standard error of the best.
-   **Kriging Metamodelling:** Anisotropic Matérn/Gaussian kernels with a learned noise term, kernel choice by leave-one-out error and LOO residual diagnostics, following Malkiel et al. (2026).
-   **Varying Scatter and Non-Gaussian Errors:** Models how the scatter changes across the inputs with a kernel smoother, and picks the error distribution (e.g. Normal, Gumbel, Logistic) by AIC.
-   **Uncertainty Quantification:** Bootstrap resampling gives confidence bounds at several levels at once, and a full $a_{X/Y}$ reliability matrix (e.g. $a_{90/95}$).
-   **Sensitivity Analysis:** Total-order Sobol indices show which inputs drive the signal.
-   **Classical Comparison:** `linear_pod()` runs the classical $\hat{a}$-versus-$a$ analysis on the same data for side-by-side comparison.

### 5. Speed

-   **Layered caching:** Models are fitted once, so changing a threshold or slice afterwards is near-instant.
-   **C++ acceleration:** The kernel smoother and Monte Carlo integration run in a multi-threaded C++ extension, with an automatic NumPy fallback.

### 6. Desktop and Browser App

The full workflow, from experimental design to UQ, is also available as a point-and-click app: a Windows/macOS desktop application, or `uvx digiqual` from a terminal.



## Development

If you want to contribute to digiqual or run the test suite locally, follow these steps.

1.  Clone and Install

This project uses uv for dependency management.

``` bash
git clone https://github.com/JGIBristol/digiqual.git
cd digiqual
```

2.  Run Tests and Lint

The package includes a full test suite using pytest. Development tools are uv dependency groups, installed automatically by `uv run`.

``` bash
uv run pytest            # or: just test  (the memory stress test is opt-in: just test_stress)
just lint                # ruff
```

3.  Build Documentation

To preview the documentation site locally:

``` bash
just preview
```

4.  Work on the App

The GUI is the `digiqual.gui` subpackage (`src/digiqual/gui/`); `app/` only holds the Briefcase configuration for the Windows/macOS installers. Run it with `just app` (desktop window) or `just app_dev` (browser, live reload). See [Desktop App Architecture](https://jgibristol.github.io/digiqual/docs/app_architecture.html) for how it fits together and how the installers are built.

## References

**Malkiel, N., Croxford, A. J., & Wilcox, P. D. (2025).** A generalized method for the reliability assessment of safety–critical inspection. Proceedings of the Royal Society A, 481: 20240654. https://doi.org/10.1098/rspa.2024.0654

**Malkiel, N., Croxford, A. J., & Wilcox, P. D. (2026).** A comprehensive investigation of flexible and multi-dimensional simulation-based PoD analysis. NDT & E International, 159: 103596. https://doi.org/10.1016/j.ndteint.2025.103596
