#!/bin/bash

# Python Code Formatter Script
# Runs ruff with auto-fix and formats Python code in a directory

set -e  # Exit on any error

# Default directory is current directory
TARGET_DIR="${1:-.}"

# Check if target directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory '$TARGET_DIR' does not exist"
    exit 1
fi

echo "🔧 Formatting Python code in: $TARGET_DIR"
echo "----------------------------------------"

# Check if ruff is installed
if ! command -v ruff &> /dev/null; then
    echo "❌ Error: ruff is not installed"
    echo "Install with: pip install ruff"
    exit 1
fi

# Run ruff check with auto-fix
echo "🔍 Running ruff check with auto-fix..."
if ruff check "$TARGET_DIR" --fix; then
    echo "✅ Ruff check completed"
else
    echo "⚠️  Ruff found issues that couldn't be auto-fixed"
fi

# Run ruff format
echo "🎨 Running ruff format..."
if ruff format "$TARGET_DIR"; then
    echo "✅ Ruff format completed"
else
    echo "❌ Ruff format failed"
    exit 1
fi

# Remove .pyc files and __pycache__ directories
echo "🧹 Cleaning up .pyc files and __pycache__ directories..."
find "$TARGET_DIR" -type f -name "*.pyc" -delete
find "$TARGET_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "✅ Cleanup completed"

echo "----------------------------------------"
echo "🎉 Code formatting and cleanup complete!"