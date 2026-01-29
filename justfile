# Project Configuration
package_name := "digiqual"

# ----------------------------------------------------------------------------
# DEFAULT COMMAND
# ----------------------------------------------------------------------------

default:
    @just --list

# ----------------------------------------------------------------------------
# DAILY DEVELOPMENT
# ----------------------------------------------------------------------------

# Sync dependencies and set up the dev environment
install:
    uv sync --extra dev

# Run unit tests (Fast check)
test:
    uv run pytest

# Preview documentation in the browser
docs:
    uv run mkdocs serve --open

# ----------------------------------------------------------------------------
# QUALITY GATES
# ----------------------------------------------------------------------------

# The "CI Simulator": Runs Tests, Builds Package, then Cleans up
check: test
    @echo "1. Tests Passed. Now checking Docs..."
    uv run mkdocs build --strict
    @echo "2. Docs Valid. Now verifying Package Build..."
    uv build
    @echo "3. Build Successful. Cleaning up artifacts..."
    just clean
    @echo "✅ All checks passed & workspace cleaned."

# Clean up build artifacts (dist, site, caches)
clean:
    rm -rf dist/ site/ .pytest_cache/ .ruff_cache/
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type d -name "*.egg-info" -exec rm -rf {} +

# ----------------------------------------------------------------------------
# WORKFLOW
# ----------------------------------------------------------------------------

# 1. BUMP VERSION (Run this on 'dev' for features, or 'main' for hotfixes)
# Usage: 'just bump minor' (dev) or 'just bump patch' (main)
bump part="patch": check
    #!/usr/bin/env bash
    set -e

    current_branch=$(git branch --show-current)
    echo "Bumping {{part}} version on branch: $current_branch..."

    # 1. Update files
    version=$(python3 scripts/bump_version.py {{part}})
    echo "New version: $version"

    # 2. Sync lock
    uv lock

    # 3. Commit
    git add pyproject.toml uv.lock README.md docs/index.md src/digiqual/__init__.py
    git commit -m "gh-action: bump version to v$version"
    git push origin HEAD

    echo "✅ Version updated."
    if [ "$current_branch" != "main" ]; then
        echo "👉 Now go to GitHub and open a Pull Request to merge '$current_branch' into 'main'."
    else
        echo "👉 You are on main. You can now run 'just tag' to release."
    fi

# 2. PUBLISH (Run this on 'main' after merging)
tag:
    #!/usr/bin/env bash
    set -e

    # Safety: Must be on main
    current_branch=$(git branch --show-current)
    if [ "$current_branch" != "main" ]; then
        echo "❌ Error: You must be on 'main' to tag a release."
        exit 1
    fi

    # Sync
    git pull origin main

    # Tag & Push
    version=$(grep 'version =' pyproject.toml | sed -E 's/version = "(.*)"/\1/')
    echo "🚀 Tagging release v$version..."
    git tag v$version
    git push origin v$version
    echo "✅ Released! Check GitHub Actions."
