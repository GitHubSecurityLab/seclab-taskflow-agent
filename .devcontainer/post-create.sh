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

# If running in Codespaces, check for necessary secrets and print error if missing
if [ -n "$CODESPACES" ]; then
    echo "🔐 Running in Codespaces - injecting secrets from Codespaces settings..."
    if [ -n "$COPILOT_TOKEN" ]; then
        echo "Running in Codespaces - please add COPILOT_TOKEN to your Codespaces secrets"
    fi
    if [ -n "$GITHUB_AUTH_HEADER" ]; then
        echo "Running in Codespaces - please add GITHUB_AUTH_HEADER to your Codespaces secrets"
    fi
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env template..."
    cat > .env << 'EOF'

# Optional: CodeQL database base path
CODEQL_DBS_BASE_PATH=/workspaces/seclab-taskflow-agent/my_data

EOF
    echo "⚠️  Please configure the enviroment or your .env file with required tokens!"
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Create optional data directories
mkdir -p my_data

echo "✅ Development environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "Configure your environment with COPILOT_TOKEN and GITHUB_AUTH_HEADER as needed."
echo "💡 Remember to activate the virtual environment: source .venv/bin/activate"
