#!/usr/bin/env bash
#
# Build script for macOS
# Creates a single-file DictaPilot binary (not .app bundle for simplicity)
#

set -e

echo "=== DictaPilot macOS Build ==="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found. Please install Python 3.10+"
    exit 1
fi

# Install PyInstaller if needed
echo "Ensuring PyInstaller is installed..."
pip3 install pyinstaller --quiet 2>/dev/null || pip install pyinstaller --quiet

# Clean previous build
if [ -d "dist" ]; then
    echo "Cleaning previous build..."
    rm -rf dist
fi

# Build
echo "Building DictaPilot..."
pyinstaller \
    --onefile \
    --name DictaPilot \
    --console \
    --clean \
    --noconfirm \
    --distpath "dist" \
    --workpath "build" \
    --specpath "packaging" \
    --collect-all sounddevice \
    --collect-all soundfile \
    --collect-all groq \
    --collect-all keyboard \
    --collect-all pynput \
    --collect-all dotenv \
    --hidden-import sounddevice \
    --hidden-import soundfile \
    --hidden-import groq \
    --hidden-import keyboard \
    --hidden-import pynput \
    --hidden-import pynput.keyboard \
    --hidden-import pynput.mouse \
    --hidden-import dotenv \
    app.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=== Build Complete ===" 
    echo "Output: dist/DictaPilot"
    echo ""
    echo "To create release zip:"
    echo "  cd dist && zip -r DictaPilot-macos-x64.zip DictaPilot && cd .."
else
    echo "Build failed!"
    exit 1
fi
