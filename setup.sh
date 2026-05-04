#!/bin/bash

echo "Setting up COMBINED_SPEAKER_MODEL environment..."

# Switch to system Python to avoid pyenv issues
pyenv global system

# Install required packages
echo "Installing required packages..."
python3 -m pip install -r requirements.txt

echo "Setup complete! You can now run:"
echo "  python3 audio_database_manager.py"
echo "  streamlit run streamlit_ui.py"
