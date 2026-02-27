# digiqual

**Statistical Toolkit for Reliability Assessment in NDT**

`digiqual` is a Python library designed for Non-Destructive Evaluation (NDE) engineers. It implements the **Generalised** $\hat{a}$-versus-a Method, allowing users to perform reliability assessments without the rigid assumptions of linearity or constant variance found in standard methods.

> **Documentation:** [Read the full documentation here](https://jgibristol.github.io/digiqual/)


## Installation

You can install `digiqual` directly from PyPI.

### Option 1: Install via uv (Recommended)

If you are managing a project with `uv`, add `digiqual` as a dependency:
```bash
# To install the latest stable release (v0.12.0):

uv add digiqual

# To install the latest development version (main branch from github):

uv add "digiqual @ git+https://github.com/JGIBristol/digiqual.git"
```

If you just want to install it into a virtual environment without modifying a project file (e.g., for a quick script), use pip interface:

```bash
uv pip install digiqual
```

### Option 2: Install via standard pip

To install the latest stable release (v0.12.0):

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

- **Latin Hypercube Sampling (LHS):** Generate space-filling experimental designs to cover your deterministic parameter space (e.g., defect size) and stochastic nuisance parameters (e.g., roughness, orientation).
- **Scale & Bound:** Automatically scale samples to your specific variable bounds.

### 2. Data Validation & Diagnostics

Ensure your simulation outputs are statistically valid before processing.

- **Sanity Checks:** Detects overlap between variables, type errors, and insufficient sample sizes.
- **Sufficiency Diagnostics:** rigorous statistical tests to flag issues like "Input Coverage Gaps" or "Model Instability" before you trust the results.

### 3. Adaptive Refinement (Active Learning)

`digiqual` closes the loop between analysis and design.

- Smart Refinement: Use `refine()` to identify specific weaknesses in your data. It uses bootstrap committees to find regions of high uncertainty and suggests new points exactly where the model is "confused".

- Automated Workflows: Use the `optimise()` method to run a fully automated "Active Learning" loop. It generates an initial design, executes your external solver, checks diagnostics, and iteratively refines the model until statistical requirements are met.

### 4. Generalised Reliability Analysis

The package includes a full statistical engine for calculating Probability of Detection (PoD) curves.

-   **Relaxed Assumptions:** Moves beyond the rigid constraints of the classical $\hat{a}$-versus-$a$ method by handling non-linear signal responses and heteroscedastic noise.
-   **Robust Statistics:** Automatically selects the best polynomial degree and error distribution (e.g., Normal, Gumbel, Logistic) based on data fit (AIC).
-   **Uncertainty Quantification:** Uses bootstrap resampling to generate robust confidence bounds and $a_{90/95}$ estimates.



## Development

If you want to contribute to digiqual or run the test suite locally, follow these steps.

1.  Clone and Install

This project uses uv for dependency management.

``` bash
git clone https://github.com/JGIBristol/digiqual.git
cd digiqual
```

2.  Run Tests

The package includes a full test suite using pytest.

``` bash
uv run pytest
```

3.  Build Documentation

To preview the documentation site locally:

``` bash
uv run quarto preview
```

## References

**Malkiel, N., Croxford, A. J., & Wilcox, P. D. (2025).** A generalized method for the reliability assessment of safety–critical inspection. Proceedings of the Royal Society A, 481: 20240654. https://doi.org/10.1098/rspa.2024.0654
