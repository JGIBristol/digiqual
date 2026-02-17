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

# Launches the app locally from the package source
app:
    uv run python -c "import digiqual; digiqual.dq_ui()"

# --- BUILD ---
# Cleans old artifacts then creates .whl and .tar.gz files
build_package: clean
    # 1. Create the package storage folder
    mkdir -p package
    # 2. Run the standard uv build
    uv build
    # 3. Move the .whl and .tar.gz into the package folder
    # We use -f to overwrite any older versions sitting there
    mv -f dist/*.whl package/
    mv -f dist/*.tar.gz package/
    # 4. Clean up the now-empty root dist folder so PyInstaller has a fresh start
    rm -rf dist

# Cleans old artifacts then creates .app file
build_app: clean
    uv run --extra dev pyinstaller --name "Digiqual" \
    --noconfirm \
    --windowed \
    --paths="src" \
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
    # 2. Prepare the destination folder
    mkdir -p app/dist
    # 3. Clean destination (Delete old app so we don't merge/corrupt it)
    rm -rf app/dist/Digiqual.app
    # 4. Move the fresh App and Spec file
    mv dist/Digiqual.app app/dist/
    mv Digiqual.spec app/
    # 5. Nuclear Cleanup (Remove the root build folders)
    rm -rf dist build app/Digiqual.spec

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

# --- VERSIONING ---

bump part="patch":
    python3 scripts/bump_version.py {{part}}
    uv lock
    @echo "Version updated locally. Now commit and push to main."


# --- UTILS ---

# Clears the terminal screen for a fresh start
cls: clean
    @clear

# Removes all generated artifacts to keep the workspace pristine
clean:
    rm -rf _site/ api_reference/ .pytest_cache/ .ruff_cache/ .quarto objects.json _sidebar.yml docs/*.csv **/*.spec
    find . -type d -name "__pycache__" -exec rm -rf {} +
