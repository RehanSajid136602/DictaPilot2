#!/usr/bin/env pwsh
<#
# Build script for Windows
# Creates a single-file DictaPilot.exe
#

$ErrorActionPreference = "Stop"

Write-Host "=== DictaPilot Windows Build ===" -ForegroundColor Cyan

# Check Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python not found. Please install Python 3.10+"
}

# Install PyInstaller if needed
Write-Host "Ensuring PyInstaller is installed..."
pip install pyinstaller --quiet

# Clean previous build
if (Test-Path "dist") {
    Write-Host "Cleaning previous build..."
    Remove-Item -Recurse -Force "dist" -ErrorAction SilentlyContinue
}

# Build
Write-Host "Building DictaPilot.exe..."
pyinstaller `
    --onefile `
    --name DictaPilot `
    --console `
    --clean `
    --noconfirm `
    --distpath "dist" `
    --workpath "build" `
    --specpath "packaging" `
    --collect-all sounddevice `
    --collect-all soundfile `
    --collect-all groq `
    --collect-all keyboard `
    --collect-all pynput `
    --collect-all dotenv `
    --hidden-import sounddevice `
    --hidden-import soundfile `
    --hidden-import groq `
    --hidden-import keyboard `
    --hidden-import pynput `
    --hidden-import pynput.keyboard `
    --hidden-import pynput.mouse `
    --hidden-import dotenv `
    app.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n=== Build Complete ===" -ForegroundColor Green
    Write-Host "Output: dist/DictaPilot.exe" -ForegroundColor Green
    Write-Host "`nTo create release zip:"
    Write-Host "  Compress-Archive -Path dist/DictaPilot.exe -DestinationPath dist/DictaPilot-windows-x64.zip" -ForegroundColor Yellow
} else {
    Write-Error "Build failed!"
}
