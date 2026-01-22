.PHONY: help install test docs build clean clean-all check

# ----------------------------------------------------------------------------
# PROJECT CONFIGURATION
# ----------------------------------------------------------------------------
PACKAGE_NAME = digiqual

# ----------------------------------------------------------------------------
# COMMANDS
# ----------------------------------------------------------------------------

help:  ## Show this help menu
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Sync dependencies and set up the dev environment
	uv sync --extra dev

test: ## Run the full test suite with pytest
	uv run pytest

docs: ## Build and serve documentation (opens in browser)
	uv run mkdocs serve --open

deploy-docs: ## Force deploy documentation to GitHub Pages immediately
	uv run mkdocs gh-deploy --force

build: ## Build the distribution package (wheel & sdist)
	uv build

check: test ## Run tests and verify docs build (CI Simulation)
	uv run mkdocs build --strict

clean: ## Remove build artifacts and temporary files
	rm -rf dist/
	rm -rf site/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +

clean-all: clean ## deep clean: Remove the virtual environment too
	rm -rf .venv/


# ----------------------------------------------------------------------------
# RELEASING
# ----------------------------------------------------------------------------

release-patch: test ## Bump patch version (0.1.0 -> 0.1.1) and push
	@echo "Bumping patch version..."
	$(eval NEW_VER := $(shell python3 scripts/bump_version.py patch))
	@echo "New version: $(NEW_VER)"
	uv lock
	git add pyproject.toml uv.lock README.md docs/index.md
	git commit -m "chore: release v$(NEW_VER)"
	git tag v$(NEW_VER)
	git push origin main
	git push origin v$(NEW_VER)

release-minor: test ## Bump minor version (0.1.0 -> 0.2.0) and push
	@echo "Bumping minor version..."
	$(eval NEW_VER := $(shell python3 scripts/bump_version.py minor))
	@echo "New version: $(NEW_VER)"
	uv lock
	git add pyproject.toml uv.lock README.md docs/index.md
	git commit -m "chore: release v$(NEW_VER)"
	git tag v$(NEW_VER)
	git push origin main
	git push origin v$(NEW_VER)

release-major: test ## Bump major version (0.1.0 -> 1.0.0) and push
	@echo "Bumping major version..."
	$(eval NEW_VER := $(shell python3 scripts/bump_version.py major))
	@echo "New version: $(NEW_VER)"
	uv lock
	git add pyproject.toml uv.lock README.md docs/index.md
	git commit -m "chore: release v$(NEW_VER)"
	git tag v$(NEW_VER)
	git push origin main
	git push origin v$(NEW_VER)
