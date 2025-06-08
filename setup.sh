#!/bin/bash

# ==============================================
# Load environment variables from .env file
# ==============================================

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Error: .env file not found!"
    echo "Please create a .env file with your API keys. See .env.example for format."
    exit 1
fi

# Load environment variables from .env file
echo "Loading environment variables from .env file..."
set -a  # Automatically export all variables
source .env
set +a  # Stop automatically exporting

# Validate required API keys
if [ -z "$WANDB_API_KEY" ] || [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "Error: Missing required API keys in .env file"
    echo "Required variables:"
    echo "  WANDB_API_KEY=your-wandb-api-key"
    echo "  HUGGINGFACE_TOKEN=your-huggingface-token"
    echo ""
    echo "Please check your .env file and ensure both keys are set."
    exit 1
fi

echo "✅ API keys loaded successfully from .env file"

# ==============================================
# Package Manager Installation
# ==============================================

echo "Installing uv package manager..."
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH for the current session
export PATH="$HOME/.cargo/bin:$PATH"

# ==============================================
# Platform Authentication
# ==============================================

echo "Setting up platform authentication..."

echo "🔐 Logging into Weights & Biases..."
pip install wandb && wandb login $WANDB_API_KEY

echo "🔐 Logging into Hugging Face..."
pip install -U "huggingface_hub[cli]"
huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential

# ==============================================
# Dataset Download
# ==============================================

echo "📥 Downloading ImageNet dataset..."
uv run python data/download_imagenet.py

echo "🎉 Setup completed successfully!" 