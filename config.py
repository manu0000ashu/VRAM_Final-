#!/usr/bin/env python3
"""
Configuration Module
Central configuration settings for the speaker recognition system
"""

from pathlib import Path

# Audio processing settings
AUDIO_SETTINGS = {
    'sample_rate': 16000,
    'n_mfcc': 13,
    'n_fft': 512,
    'hop_length': 160,
    'n_mels': 40,
    'fmin': 0,
    'fmax': 8000
}

# Model settings
MODEL_SETTINGS = {
    'n_estimators': 100,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42,
    'test_size': 0.2
}

# File paths
PATHS = {
    'uploads': Path("uploads"),
    'mfcc_output': Path("mfcc_output"),
    'trained_model': Path("trained_simple_model"),
    'dataset_train': Path("dataset_train"),
    'dataset_test': Path("dataset_test"),
    'speaker_data': Path("speaker_dataset/50_speakers_audio_data")
}

# Supported audio formats
AUDIO_FORMATS = {'wav', 'mp3', 'flac', 'm4a', 'ogg'}

# UI settings
UI_SETTINGS = {
    'confidence_threshold': 0.7,
    'max_upload_size': 50 * 1024 * 1024,  # 50MB
    'session_timeout': 3600  # 1 hour
}

# Visualization settings
VIZ_SETTINGS = {
    'figure_size': (12, 8),
    'dpi': 150,
    'color_maps': {
        'mfcc': 'viridis',
        'delta': 'plasma', 
        'delta2': 'coolwarm'
    }
}
