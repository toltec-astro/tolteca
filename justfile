# Justfile for tolteca

# Show available commands
list:
    @just --list

install:
    uv sync --all-groups --all-packages --all-extras
    git config core.hooksPath .githooks
    uv run pre-commit install-hooks

rebuild-lockfiles:
    uv lock --upgrade

# Run all the formatting, linting, and testing commands
qa:
    uv run pre-commit run --all-files
    uv run coverage run -m pytest
    just doc-test

# Run tests with coverage
coverage:
    uv run coverage run -m pytest
    uv run coverage combine
    uv run coverage report -m
    uv run coverage html

# Build the project, useful for checking that packaging is correct
build:
    rm -rf build
    rm -rf dist
    uv build

# Publish to PyPI (manual alternative to GitHub Actions)
publish: build
    uv publish

clean:
    rm -fr build/
    rm -fr dist/
    rm -fr docs/_build/
    rm -fr docs/api/
    rm -fr .eggs/
    find . -name '*.egg-info' -exec rm -fr {} +
    find . -name '*.egg' -exec rm -fr {} +
    find . -name '*.pyc' -exec rm -f {} +
    find . -name '*.pyo' -exec rm -f {} +
    find . -name '*~' -exec rm -f {} +
    find . -name '__pycache__' -exec rm -fr {} +
    rm -f .coverage
    rm -fr htmlcov/
    rm -fr .pytest_cache
    rm -fr .ruff_cache

# The docs link to Tollan's unpublished API: build Tollan's docs first, or set
# TOLLAN_DOCS_INVENTORY. Tolteca's dependencies are workspace members, so the
# docs run in the deployment location environment (load its .envrc).
tollan_inventory := env("TOLLAN_DOCS_INVENTORY", justfile_directory() + "/../tollan/docs/_build/html/objects.inv")
sphinx := '"${UV_PROJECT_ENVIRONMENT:?Load the deployment location .envrc}"/bin/sphinx-build'

# Build the docs
doc-build:
    rm -rf docs/api
    TOLLAN_DOCS_INVENTORY={{ quote(tollan_inventory) }} {{ sphinx }} -M html docs docs/_build -E -a -n -W --keep-going -T

# Execute guide examples with Sphinx
doc-test:
    rm -rf docs/api
    TOLLAN_DOCS_INVENTORY={{ quote(tollan_inventory) }} {{ sphinx }} -M doctest docs docs/_build -E -a -n -W --keep-going -T

# Serve docs locally with live reload
doc:
    uv run --group docs sphinx-autobuild docs docs/_build/html --port 8888 --open-browser

# Deploy docs to GitHub Pages
doc-deploy: doc-build
    uv run --group docs ghp-import docs/_build/html -r origin -b gh-pages --push --no-jekyll

# Check if project is up to date with the template
# NOTE: --checkout must be passed explicitly; cruft does not read it from .cruft.json (upstream issue)
cruft-check:
    uvx cruft check --checkout v2026

# Update project from the template
cruft-update:
    uvx cruft update --checkout v2026
