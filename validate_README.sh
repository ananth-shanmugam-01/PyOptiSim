#!/bin/bash
set -e  # Exit on error

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade packaging tools
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt

# Run example
python Example_OCP.py
echo "README steps validated successfully!"