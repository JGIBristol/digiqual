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
# RELEASING
# ----------------------------------------------------------------------------

# 🚀 Bump version & Push. (Usage: just release minor)
# SAFETY: Automatically runs 'check' first. If 'check' fails, this aborts.
release part="patch": check
    #!/usr/bin/env bash
    set -e
    echo "Bumping {{part}} version..."

    # 1. Update version in files
    # (The script runs, updates files, and we capture the version number)
    version=$(python3 scripts/bump_version.py {{part}})
    echo "New version: $version"

    # 2. Update lock file
    uv lock

    # 3. Commit, Tag, Push
    git add pyproject.toml uv.lock README.md docs/index.md
    git commit -m "gh-action: release v$version"
    git tag v$version
    git push origin main
    git push origin v$version

# ----------------------------------------------------------------------------
# DOCS HOTFIX (Use when you find a typo on the live site)
# ----------------------------------------------------------------------------

# 1. Start a hotfix: Jumps to the latest tag so you can edit the live docs
fix-docs:
    @if [ -n "$(git status --porcelain)" ]; then echo "❌ Work tree is dirty. Please commit or stash changes first."; exit 1; fi
    @echo "Finding latest tag..."
    tag=$(git describe --tags --abbrev=0); \
    echo "Checking out $tag..."; \
    git checkout $tag; \
    echo "✅ You are now editing version $tag."; \
    echo "👉 Make your changes now. When done, run 'just finish-fix'"

# 2. Finish hotfix: Deploys the site and returns you to main
finish-fix:
    @echo "Deploying fixed documentation..."
    uv run mkdocs gh-deploy --force
    @echo "Stashing changes to carry them to main..."
    git stash
    @echo "Returning to main branch..."
    git checkout main
    @echo "Restoring changes..."
    git stash pop
    @echo "✅ Back on main. Your typo fix is in your working directory."
    @echo "👉 Don't forget to commit the fix to main!"
