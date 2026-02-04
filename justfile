package_name := "digiqual"

default:
    @just --list

# --- DEVELOPMENT ---

# Installs all relevant packages
install:
    uv sync --extra dev

# Runs pytest
test:
    uv run pytest

# --- BUILD ---
# Cleans old artifacts then creates .whl and .tar.gz files
build: clean
    uv build

# --- DOCUMENTATION ---
# Preview Website
preview:
    uv run quartodoc build
    uv run quarto preview index.qmd

# Manually pushes to the gh-pages branch without a CI logjam
publish: clean
    uv run quartodoc build
    uv run quarto publish gh-pages --no-prompt
    clean

# --- VERSIONING ---

bump part="patch":
    python3 scripts/bump_version.py {{part}}
    uv lock
    @echo "Version updated locally. Now commit and push to main."

# --- UTILS ---

# Removes all generated artifacts to keep the workspace pristine
clean:
    rm -rf dist/ _site/ api_reference/ .pytest_cache/ .ruff_cache/ .quarto objects.json _sidebar.yml
    find . -type d -name "__pycache__" -exec rm -rf {} +
