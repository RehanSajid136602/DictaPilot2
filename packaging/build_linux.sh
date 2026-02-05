#!/usr/bin/env bash
#
# Build script for Linux
# Creates a single-file DictaPilot binary
# Optionally creates AppImage
#

set -e

echo "=== DictaPilot Linux Build ==="

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
    
    # Optionally create AppImage
    if [ "$1" = "--appimage" ] || [ "$2" = "--appimage" ]; then
        echo "Creating AppImage..."
        create_appimage
    else
        echo "To create AppImage, run: ./packaging/build_linux.sh --appimage"
        echo "To create .deb, run: ./packaging/build_deb.sh"
    fi
else
    echo "Build failed!"
    exit 1
fi

create_appimage() {
    # Check for appimagetool
    if ! command -v appimagetool &> /dev/null; then
        echo "Error: appimagetool not found."
        echo "Download from: https://github.com/AppImage/AppImageKit/releases"
        echo "Or install via: curl -L https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage -o /usr/local/bin/appimagetool && chmod +x /usr/local/bin/appimagetool"
        return 1
    fi
    
    # Create AppDir structure
    APPDIR="DictaPilot-x86_64.AppDir"
    mkdir -p "$APPDIR/usr/bin"
    mkdir -p "$APPDIR/usr/share/applications"
    mkdir -p "$APPDIR/usr/share/icons/hicolor/256x256/apps"
    
    # Copy binary
    cp dist/DictaPilot "$APPDIR/usr/bin/"
    chmod +x "$APPDIR/usr/bin/DictaPilot"
    
    # Create .desktop file
    cat > "$APPDIR/usr/share/applications/dictapilot.desktop" << 'EOF'
[Desktop Entry]
Name=DictaPilot
Comment=Cross-platform dictation app
Exec=DictaPilot %U
Icon=dictapilot
Terminal=false
Type=Application
Categories=Office;Utility;
EOF
    
    # Copy icon (create simple placeholder if none exists)
    if [ ! -f "packaging/icon.png" ]; then
        # Create a simple 256x256 placeholder PNG
        python3 -c "
from PIL import Image, ImageDraw
img = Image.new('RGB', (256, 256), color='#1a1a2e')
draw = ImageDraw.Draw(img)
draw.ellipse([20, 20, 236, 236], fill='#4a4a6a')
draw.text([80, 100], 'DP', fill='white')
img.save('$APPDIR/usr/share/icons/hicolor/256x256/apps/dictapilot.png')
" 2>/dev/null || echo "Note: Could not create icon placeholder"
    else
        cp packaging/icon.png "$APPDIR/usr/share/icons/hicolor/256x256/apps/dictapilot.png"
    fi
    
    # Copy AppImage metadata
    cat > "$APPDIR/AppImage.toml" << 'EOF'
[AppImage]
Name=DictaPilot
ID=com.dictapilot.app
Version=1.0.0
Arch=x86_64
EOF
    
    # Generate AppImage
    appimagetool "$APPDIR" dist/DictaPilot-x86_64.AppImage
    
    echo "AppImage created: dist/DictaPilot-x86_64.AppImage"
}
