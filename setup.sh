#!/bin/bash

# ==============================================
# Load environment variables from .env file
# ==============================================
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' .env | xargs)
else
    echo "Error: .env file not found. Please create a .env file with your API keys."
    echo "Example .env file:"
    echo "HUGGINGFACE_TOKEN=your-huggingface-token-here"
    echo "WANDB_API_KEY=your-wandb-api-key-here"
    exit 1
fi

# Validate API keys
if [ -z "$HUGGINGFACE_TOKEN" ] || [ -z "$WANDB_API_KEY" ]; then
    echo "Error: Missing required environment variables in .env file"
    echo "Please ensure your .env file contains:"
    echo "HUGGINGFACE_TOKEN=your-huggingface-token-here"
    echo "WANDB_API_KEY=your-wandb-api-key-here"
    exit 1
fi

# echo "Installing uv package manager..."
# curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH for the current session
export PATH="$HOME/.cargo/bin:$PATH"

echo "Setting up platform authentication..."
# Set WANDB_API_KEY environment variable (no interactive login needed)
export WANDB_API_KEY

# Install and login to HuggingFace non-interactively
uv pip install -U "huggingface_hub[cli]"
huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential

echo "Setup completed successfully!"
