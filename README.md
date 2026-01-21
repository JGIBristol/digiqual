# DigiQual

**Statistical Toolkit for Reliability Assessment in NDT**

`digiqual` is a Python library designed for Non-Destructive Evaluation (NDE) engineers. It implements the **Generalized** $\hat{a}$-versus-a Method, allowing users to perform reliability assessments without the rigid assumptions of linearity or constant variance found in standard methods.

> **Documentation:** [Read the full documentation here](https://github.com/JGIBristol/DigiQual-Python)

------------------------------------------------------------------------

## Installation

You can install `digiqual` directly from PyPI (once published) or from source.

### Using pip

``` bash
pip install digiqual
```

### Using uv (Recommended)

``` bash
uv pip install digiqual
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
git clone [https://github.com/JGIBristol/DigiQual-Python.git](https://github.com/JGIBristol/DigiQual-Python.git)
cd digiqual
uv pip install -e ".[dev]"  # Install package in editable mode + dev tools
```

2.  Run Tests

The package includes a full test suite using pytest.

``` bash
pytest
```

3.  Build Documentation

To preview the documentation site locally:

``` bash
mkdocs serve
```

## References

**Malkiel, N., Croxford, A. J., & Wilcox, P. D. (2025).** A generalized method for the reliability assessment of safety–critical inspection. Proceedings of the Royal Society A, 481: 20240654. https://doi.org/10.1098/rspa.2024.0654
