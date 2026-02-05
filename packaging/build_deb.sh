#!/usr/bin/env bash
#
# Build script for .deb package
# Requires fpm (Effing Package Management)
#

set -e

echo "=== DictaPilot .deb Build ==="

# Check for fpm
if ! command -v fpm &> /dev/null; then
    echo "Error: fpm not found."
    echo "Install with: gem install fpm"
    echo "Or: sudo apt install ruby-dev && sudo gem install fpm"
    exit 1
fi

# Check if binary exists
if [ ! -f "dist/DictaPilot" ]; then
    echo "Error: dist/DictaPilot not found. Run build_linux.sh first."
    exit 1
fi

# Clean previous package
rm -f dist/dictapilot_*.deb

# Get version from git or use 1.0.0
VERSION=$(git describe --tags --always 2>/dev/null | sed 's/v//') || VERSION="1.0.0"

echo "Building package version: $VERSION"

# Create temporary staging directory
STAGING=$(mktemp -d)
mkdir -p "$STAGING/usr/bin"
mkdir -p "$STAGING/usr/share/applications"
mkdir -p "$STAGING/usr/share/icons/hicolor/256x256/apps"

# Copy binary
cp dist/DictaPilot "$STAGING/usr/bin/"
chmod 755 "$STAGING/usr/bin/DictaPilot"

# Create .desktop file
cat > "$STAGING/usr/share/applications/dictapilot.desktop" << 'EOF'
[Desktop Entry]
Name=DictaPilot
Comment=Cross-platform dictation app with smart editing
Exec=DictaPilot %U
Icon=dictapilot
Terminal=false
Type=Application
Categories=Office;Utility;
EOF

# Create icon placeholder (simple Python script)
python3 -c "
from PIL import Image, ImageDraw
img = Image.new('RGB', (256, 256), color='#1a1a2e')
draw = ImageDraw.Draw(img)
draw.ellipse([20, 20, 236, 236], fill='#4a4a6a')
draw.text([80, 100], 'DP', fill='white')
img.save('$STAGING/usr/share/icons/hicolor/256x256/apps/dictapilot.png')
" 2>/dev/null || echo "Note: Could not create icon"

# Create postinst script
cat > "$STAGING/postinst" << 'EOF'
#!/bin/bash
set -e
if [ "$1" = "configure" ] || [ "$1" = "1" ]; then
    chmod 4755 /usr/bin/DictaPilot 2>/dev/null || true
fi
exit 0
EOF
chmod 755 "$STAGING/postinst"

# Build .deb
fpm \
    --input-type dir \
    --output-type deb \
    --name dictapilot \
    --version "$VERSION" \
    --architecture amd64 \
    --maintainer "Rehan <rehan@example.com>" \
    --description "Cross-platform dictation app with smart editing" \
    --url "https://github.com/RehanSajid136602/DictaPilot" \
    --license MIT \
    --depends python3 \
    --depends python3-pip \
    --depends libportaudio2 \
    --after-install "$STAGING/postinst" \
    -C "$STAGING" .

# Cleanup
rm -rf "$STAGING"

if [ -f "dictapilot_${VERSION}_amd64.deb" ]; then
    mv dictapilot_${VERSION}_amd64.deb dist/
    echo ""
    echo "=== .deb Build Complete ===" 
    echo "Output: dist/dictapilot_${VERSION}_amd64.deb"
else
    echo "Build failed!"
    exit 1
fi
