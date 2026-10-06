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

# Runs pytest across all supported Python versions (3.11 to 3.14)
test_matrix:
    #!/usr/bin/env bash
    set -euo pipefail

    # Ensure Apple Silicon builds target macOS 11.0+ to prevent libc++ deprecation warnings
    export MACOSX_DEPLOYMENT_TARGET="11.0"

    for ver in 3.11 3.12 3.13 3.14; do
        echo ""
        echo "🚀 ======================================"
        echo "🧪 Testing with Python $ver..."
        echo "========================================"
        echo ""
        uv run --python "$ver" --extra dev python setup.py build_ext --inplace
        uv run --python "$ver" --extra dev pytest
    done

    echo ""
    echo "🧹 Cleaning up: Reverting .venv back to Python 3.11..."
    uv sync --python 3.11 --extra dev
    uv run python setup.py build_ext --inplace
    echo "✅ All tests passed and development environment restored!"

# Run the app in "Browser Mode" (Best for coding/debugging)
app_dev:
    cd app && uv run shiny run app.py

# Run the app in "Desktop Mode" (Best for testing the .exe look)
app:
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

# (Optional) Builds the .app bundle locally on Mac for quick dev testing
build_app_local: clean
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
    @echo "Local macOS build complete. App located at app/dist/Digiqual.app"

# Triggers the cross-platform GitHub Action workflow to build Windows & Mac app executables
trigger_build:
        gh workflow run build_app.yml && \
        echo "🚀 Cross-platform app build workflow triggered on GitHub Actions!" && \
        echo "Run 'gh run list --workflow=build_app.yml' or check GitHub UI to monitor progress."; \


# Manual/emergency-only: uploads a SINGLE locally-built wheel (this machine's
# platform only) straight to PyPI. Normal releases must NOT use this -- push a
# `vX.Y.Z` tag instead, which triggers build_wheels.yml to build real wheels
# for every supported platform in CI and publish them. This recipe bypasses
# that entirely and is only here as a last-resort fallback.
build_pypi: clean
    # uv publish takes everything in your custom package/ directory
    uv publish package/*


# --- DOCUMENTATION ---
# Preview Website
preview: clean
    uv run quartodoc build
    uv run quarto preview index.qmd

# Manually pushes to the gh-pages branch
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
    rm -rf _site/ api_reference/ .pytest_cache/ .ruff_cache/ .quarto objects.json _sidebar.yml docs/*.csv **/*.spec *.csv *.egg-info build/ dist/ app/build/ app/dist/ *.zip src/*.egg-info src/digiqual/*.so src/digiqual/*.pyd src/digiqual/*.dylib
    find . -type d -name "__pycache__" -exec rm -rf {} +


# --- COMBO ---

# Patch Combo: cleans, tests, bumps a patch version, then commits, pushes and
# tags -- the tag push triggers build_wheels.yml, which builds per-platform
# wheels in CI and publishes them to PyPI. Docs are published last and
# non-fatally, since `quarto publish`'s post-push deploy check has timed out
# before without the actual push failing -- that must never block the release.
# Does NOT create a GitHub Release; run `gh release create vX.Y.Z
# --generate-notes` yourself once the Actions run is green.
patch: clean
    #!/usr/bin/env bash
    set -euo pipefail
    just _preflight_release
    just test_matrix
    just bump patch
    just cls
    just _commit_tag_push
    just build_website || echo "WARNING: build_website failed (docs may not be published) -- the release above already completed successfully."

# Minor Combo: cleans, tests, bumps a minor version, then commits, pushes and
# tags -- the tag push triggers build_wheels.yml, which builds per-platform
# wheels in CI and publishes them to PyPI. Docs are published last and
# non-fatally, since `quarto publish`'s post-push deploy check has timed out
# before without the actual push failing -- that must never block the release.
# Does NOT create a GitHub Release; run `gh release create vX.Y.Z
# --generate-notes` yourself once the Actions run is green.
minor: clean
    #!/usr/bin/env bash
    set -euo pipefail
    just _preflight_release
    just test_matrix
    just bump minor
    just cls
    just _commit_tag_push
    just build_website || echo "WARNING: build_website failed (docs may not be published) -- the release above already completed successfully."

# Internal: refuse to start a release combo from a dirty tree or off `main`.
_preflight_release:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -n "$(git status --porcelain)" ]; then
        echo "error: working tree is not clean -- commit or stash first." >&2
        exit 1
    fi
    branch=$(git rev-parse --abbrev-ref HEAD)
    if [ "$branch" != "main" ]; then
        echo "error: release combos must run from 'main' (currently on '$branch')." >&2
        exit 1
    fi

# Internal: commits the version-bump files, pushes, then tags and pushes the tag.
_commit_tag_push:
    #!/usr/bin/env bash
    set -euo pipefail
    NEW_VERSION=$(python3 -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")
    git add pyproject.toml uv.lock README.md index.qmd src/digiqual/__init__.py docs/install.qmd app/app.py app/run_app.py setup.py
    git commit -m "Bump to v${NEW_VERSION}"
    git push
    git tag "v${NEW_VERSION}"
    git push origin "v${NEW_VERSION}"
    echo ""
    echo "Pushed commit + tag v${NEW_VERSION}. build_wheels.yml is now building wheels and will publish to PyPI:"
    echo "  https://github.com/JGIBristol/digiqual/actions"
    echo "Once that run is green, create the GitHub Release yourself with:"
    echo "  gh release create v${NEW_VERSION} --generate-notes"
