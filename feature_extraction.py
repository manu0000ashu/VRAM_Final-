#!/usr/bin/env python3
"""
Feature Extraction Module - Legacy Import Wrapper
Now imports from centralized extract_mfcc module
"""

# Import all MFCC functions from centralized module
from extract_mfcc import (
    MFCCExtractor,
    extract_mfcc_features,
    extract_features_batch,
    extract_39dim_features,
    cosine_similarity,
    extract_mfcc_for_graphs,
    extract_mfcc_for_training
)

# Import UI features directly from the class
def extract_ui_features(audio_path):
    """Extract UI features using MFCCExtractor"""
    extractor = MFCCExtractor()
    return extractor.extract_ui_features(audio_path)

# Create default extractor instance for backward compatibility
FeatureExtractor = MFCCExtractor
