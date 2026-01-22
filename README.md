# digiqual

**Statistical Toolkit for Reliability Assessment in NDT**

`digiqual` is a Python library designed for Non-Destructive Evaluation (NDE) engineers. It implements the **Generalized** $\hat{a}$-versus-a Method, allowing users to perform reliability assessments without the rigid assumptions of linearity or constant variance found in standard methods.

> **Documentation:** [Read the full documentation here](https://jgibristol.github.io/digiqual/)


## Installation

You can install `digiqual` directly from GitHub.

### Option 1: Install via uv (Recommended)

If you are managing a project with `uv`, add `digiqual` as a dependency:
```bash
# To install the latest stable release (v0.2.0):

uv add "digiqual @ git+https://github.com/JGIBristol/digiqual.git@v0.2.0"

# To install the latest development version (main branch):

uv add "digiqual @ git+https://github.com/JGIBristol/digiqual.git"
```

If you just want to install it into a virtual environment without modifying a project file (e.g., for a quick script), use pip interface:

```bash
uv pip install "git+https://github.com/JGIBristol/digiqual.git@v0.2.0"
```

### Option 2: Install via standard pip

To install the latest stable release (v0.2.0):

```bash
pip install "git+https://github.com/JGIBristol/digiqual.git@v0.2.0"
```
To install the latest development version:

```bash
pip install "git+https://github.com/JGIBristol/digiqual.git"
```

## Features

-   Experimental Design: Generate Latin Hypercube Sampling (LHS) designs for simulation inputs.

-   Data Validation: Automatically check simulation results for type errors, overlaps, and sample size sufficiency.

-   Reliability Analysis: (In Development) Calculate Probability of Detection (PoD) curves using advanced regression and bootstrap confidence bounds.


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
uv run mkdocs serve
```

## References

**Malkiel, N., Croxford, A. J., & Wilcox, P. D. (2025).** A generalized method for the reliability assessment of safety–critical inspection. Proceedings of the Royal Society A, 481: 20240654. https://doi.org/10.1098/rspa.2024.0654
