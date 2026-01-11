# Installation Guide

## Option 1: Using uv (Recommended - Faster!)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment
uv venv

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt

# Install development dependencies (optional)
uv pip install -r requirements-dev.txt

# Install the package in editable mode
uv pip install -e .
```

## Option 2: Using pip (Traditional)

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt

# Install the package in editable mode
pip install -e .
```

## Quick Start with uv

```bash
# One-liner to get started
curl -LsSf https://astral.sh/uv/install.sh | sh && \
uv venv && \
source .venv/bin/activate && \
uv pip install -r requirements.txt && \
uv pip install -e .
```

## Verify Installation

```bash
# Check that the CLI is installed
linalg-tutor --help

# Or run as a module
python -m linalg_tutor --help
```

## Running Tests

```bash
# Make sure dev dependencies are installed
uv pip install -r requirements-dev.txt

# Run tests
pytest

# Run with coverage
pytest --cov=linalg_tutor
```
