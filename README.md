# COMBINED_SPEAKER_MODEL

A speaker recognition system using a 39-dim Siamese-like Network trained on 40 speakers.

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Quick Start

### Setup Environment
```bash
./setup.sh
```

### Run Applications

**CLI Interface:**
```bash
python3 audio_database_manager.py
```

**Web Interface:**
```bash
streamlit run streamlit_ui.py
```

## Model Details

- **Architecture**: 39-dim Siamese-like Network
- **Input Features**: 39-dimensional MFCC (13 MFCC + 13 delta + 13 delta2)
- **Trained Speakers**: 40 speakers
- **Audio Processing**: 16kHz sample rate

## Dependencies

See `requirements.txt` for full list of dependencies.

## Troubleshooting

If you encounter numpy import errors:
1. Run `./setup.sh` to ensure proper environment
2. Use `python3` instead of `python` to avoid pyenv conflicts
3. Ensure system Python is being used: `pyenv global system`

## Features

- 🎵 Upload and store audio files
- 🔍 Compare audio for speaker identification
- 📊 Generate visualizations and graphs
- 🌐 Web-based interface via Streamlit
- 💾 Audio database management

## GitHub Deployment

### Clone and Run
```bash
git clone <YOUR_REPO_URL>
cd COMBINED_SPEAKER_MODEL
./setup.sh
python3 audio_database_manager.py
```

### Web Deployment (Streamlit Cloud)
1. Fork this repository
2. Connect to Streamlit Cloud
3. Set main file to `streamlit_ui.py`
4. Deploy

### Local Development
```bash
git clone <YOUR_REPO_URL>
cd COMBINED_SPEAKER_MODEL
pip install -r requirements.txt
streamlit run streamlit_ui.py
```

## Project Structure
```
COMBINED_SPEAKER_MODEL/
├── audio_database_manager.py    # CLI interface
├── streamlit_ui.py              # Web interface
├── extract_mfcc.py              # Feature extraction
├── advanced_graph_generator.py  # Visualizations
├── siamese_model_39dim/          # Trained model
├── audio_database/              # Runtime database
├── requirements.txt             # Dependencies
├── setup.sh                     # Setup script
└── README.md                    # Documentation
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
