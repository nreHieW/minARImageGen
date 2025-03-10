#!/bin/bash

# ==============================================
# Configuration - Replace with your API keys
# ==============================================
WANDB_USER_KEY="your-wandb-api-key-here"
HUGGINGFACE_USER_KEY="your-huggingface-api-key-here"
# ==============================================

# Validate API keys
if [ "$WANDB_USER_KEY" = "your-wandb-api-key-here" ] || [ "$HUGGINGFACE_USER_KEY" = "your-huggingface-api-key-here" ]; then
    echo "Error: Please replace the API key placeholders in the script with your actual API keys"
    exit 1
fi

# Export the keys
export WANDB_USER_KEY
export HUGGINGFACE_USER_KEY

echo "Installing uv package manager..."
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH for the current session
export PATH="$HOME/.cargo/bin:$PATH"

echo "Creating virtual environment and installing dependencies..."
uv venv
source .venv/bin/activate
uv pip install .

echo "Setting up platform authentication..."
pip install wandb && wandb login --verify $WANDB_USER_KEY
pip install -U "huggingface_hub[cli]"
huggingface-cli login --token $HUGGINGFACE_USER_KEY --add-to-git-credential

echo "Downloading ImageNet dataset..."
python data/download_imagenet.py

echo "Setup completed successfully!" 