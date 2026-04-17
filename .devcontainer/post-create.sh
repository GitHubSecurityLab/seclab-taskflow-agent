#!/bin/bash
set -e

echo "🚀 Setting up Seclab Taskflow Agent development environment..."

# Create Python virtual environment
echo "📦 Creating Python virtual environment..."
python3 -m venv .venv

# Activate virtual environment and install dependencies
echo "📥 Installing Python dependencies..."
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install hatch
hatch build

# Install this package from local directory.
pip install -e .

# Install codeql-development-mcp-server and its CodeQL tool query packs
echo "📦 Installing codeql-development-mcp-server..."
npm install -g codeql-development-mcp-server
codeql-development-mcp-server-setup-packs

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env template..."
    echo "# Optional: CodeQL database base path" >> .env
    echo "⚠️  Please configure the environment or your .env file with required tokens!"
fi

echo "✅ Development environment setup complete!"
