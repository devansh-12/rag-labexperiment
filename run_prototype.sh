#!/usr/bin/env bash

# Quick prototype setup script for SPIT RAG chatbot
# This script creates a virtual environment, installs dependencies,
# copies the example environment file, and starts the FastAPI server.
# It assumes you have Python 3.10+ and internet connectivity.

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Create virtual environment
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install required packages
if [ -f "requirements.txt" ]; then
  echo "Installing dependencies from requirements.txt..."
  pip install -r requirements.txt
else
  echo "ERROR: requirements.txt not found in $PROJECT_ROOT"
  exit 1
fi

# Copy example env if .env does not exist
if [ ! -f ".env" ]; then
  if [ -f ".env.example" ]; then
    cp .env.example .env
    echo "Copied .env.example to .env. Please edit .env to add your API keys (e.g., OPENAI_API_KEY or GEMINI_API_KEY)."
  else
    echo "WARNING: .env.example not found. You may need to create a .env file manually."
  fi
fi

# Export variables from .env for the current session
if [ -f ".env" ]; then
  eval "$(python3 -c '
from dotenv import dotenv_values
import shlex
env = dotenv_values(".env")
for k, v in env.items():
    if k and v is not None:
        print(f"export {k}={shlex.quote(str(v))}")
')"
fi

# Run the FastAPI app (app.py should expose a FastAPI instance named "app")
# Use --reload for development hot‑reloading.
if [ -f "app.py" ]; then
  echo "Starting FastAPI server..."
  uvicorn app:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" --reload
else
  echo "ERROR: app.py not found. Ensure you are in the project root and the file exists."
  exit 1
fi
