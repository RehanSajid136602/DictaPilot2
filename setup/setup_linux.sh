#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERROR] Python 3.10+ not found."
  echo "Install Python 3.10+ and re-run."
  read -r -p "Press Enter to close..."
  exit 1
fi

if [ ! -d "venv" ]; then
  echo "[INFO] Creating virtual environment..."
  if ! python3 -m venv venv; then
    echo "[ERROR] Failed to create venv."
    echo "On Debian/Ubuntu, run: sudo apt install python3-venv"
    read -r -p "Press Enter to close..."
    exit 1
  fi
fi

# shellcheck disable=SC1091
source venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [ ! -f ".env" ]; then
  if [ -f ".env.example" ]; then
    cp ".env.example" ".env"
    echo "[INFO] Created .env from .env.example"
  else
    echo "[WARN] .env.example not found. Create .env manually."
  fi
fi

echo
echo "Setup complete."
echo "1) Open .env and set GROQ_API_KEY"
echo "2) Run: python app.py"
echo
read -r -p "Press Enter to close..."
