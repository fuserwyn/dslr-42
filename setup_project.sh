#!/bin/bash

# Exit on error
set -e

echo "🐍 Creating virtual environment..."
python3 -m venv .venv

echo "✅ Virtual environment created."

# Activate virtualenv
source .venv/bin/activate

echo "⬇️ Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Dependencies installed."
echo "📦 To activate the virtual environment, run:"
echo "   source .venv/bin/activate"
