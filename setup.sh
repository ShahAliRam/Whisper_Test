#!/bin/bash

echo "Checking operating system..."

# Detect OS type
OS="$(uname -s)"

# Detect Python command
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
elif command -v py &> /dev/null; then
    PYTHON=py
else
    echo "Python not found! Install Python first."
    exit 1
fi

echo "Using Python: $PYTHON"

# Create virtual environment
echo "Creating virtual environment..."
$PYTHON -m venv venv

# Activate based on OS
case "$OS" in
    Linux|Darwin*)
        echo "Detected Linux or macOS, activating venv..."
        source venv/bin/activate
        ;;
    MINGW*|CYGWIN*|MSYS*)
        echo "Detected Windows (Git Bash), activating venv..."
        source venv/Scripts/activate
        ;;
    *)
        echo "Unknown OS: $OS"
        exit 1
        ;;
esac

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "requirements.txt not found, skipping install."
fi

# Run app
echo "Running app.py..."
$PYTHON app.py
