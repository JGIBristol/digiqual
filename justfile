package_name := "digiqual"

default:
    @just --list

# --- DEVELOPMENT ---

# Installs all relevant packages
sync:
    uv sync --all-extras

# Runs pytest
test:
    uv run pytest

# Runs pytest across all supported Python versions (3.10 to 3.13)
test_matrix:
    @for ver in 3.11 3.12 3.13 3.14; do \
        echo "\n🚀 ======================================"; \
        echo "🧪 Testing with Python $ver..."; \
        echo "========================================\n"; \
        uv run --python $ver --extra dev pytest || exit 1; \
    done
    @echo "\n🧹 Cleaning up: Reverting .venv back to Python 3.11..."
    uv sync --python 3.11 --extra dev
    @echo "✅ All tests passed and development environment restored!"

# Run the app in "Browser Mode" (Best for coding/debugging)
app_dev:
    cd app && uv run shiny run app.py

# Run the app in "Desktop Mode" (Best for testing the .exe look)
app_desktop:
    cd app && uv run python run_app.py

# --- VERSIONING ---

bump part="patch":
    python3 scripts/bump_version.py {{part}}
    uv lock
    @echo "Version updated locally. Now commit and push to main."



# --- BUILD ---
# Cleans old artifacts then creates .whl and .tar.gz files
build_package: clean
    # 1. Create the package storage folder (and ensure it is empty)
    rm -rf package/
    mkdir -p package
    # 2. Run the standard uv build
    uv build
    # 3. Move the .whl and .tar.gz into the package folder
    # These will now be the only files in 'package/'
    mv dist/*.whl package/
    mv dist/*.tar.gz package/
    # 4. Clean up the now-empty root dist folder
    rm -rf dist

# Cleans old artifacts then creates .app file
build_app: clean
    # 1. Enter app folder AND run pyinstaller in one chain
    # We use --directory to tell uv where to run
    cd app && uv run pyinstaller --name "Digiqual" \
        --noconfirm \
        --windowed \
        --collect-all digiqual \
        --collect-all shiny \
        --collect-all faicons \
        --collect-all shinyswatch \
        --collect-all htmltools \
        --collect-all pywebview \
        --hidden-import="uvicorn.loops.auto" \
        --hidden-import="uvicorn.protocols.http.auto" \
        --hidden-import="uvicorn.lifespan.on" \
        --hidden-import="engineio.async_drivers.threading" \
        run_app.py

    # 2. Organize the output (PyInstaller creates dist/ inside app/ now)
    # We just need to make sure the final .app is where you expect it
    @echo "Build complete. App is located at app/dist/Digiqual.app"


# Uploads the package to PyPI (bump version before)
build_pypi: clean
    # uv publish takes everything in your custom package/ directory
    uv publish package/*


# --- DOCUMENTATION ---
# Preview Website
preview: clean
    uv run quartodoc build
    uv run quarto preview index.qmd

# Manually pushes to the gh-pages branch without a CI logjam
build_website: clean
    uv run quartodoc build
    uv run quarto publish gh-pages --no-prompt
    just clean


# --- UTILS ---

# Clears the terminal screen for a fresh start
cls: clean
    @clear

# Removes all generated artifacts to keep the workspace pristine
clean:
    rm -rf _site/ api_reference/ .pytest_cache/ .ruff_cache/ .quarto objects.json _sidebar.yml docs/*.csv **/*.spec
    find . -type d -name "__pycache__" -exec rm -rf {} +
