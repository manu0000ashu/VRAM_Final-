#!/usr/bin/env python3
"""
Centralized MFCC Extraction Module
Handles all MFCC and audio feature extraction from audio files
"""

import numpy as np
import librosa
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

class MFCCExtractor:
    """Centralized MFCC and audio feature extraction"""
    
    def __init__(self, sample_rate=16000, n_mfcc=13):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        
    def extract_mfcc_features(self, audio_path, n_mfcc=13):
        """Extract MFCC features from audio file
        
        Args:
            audio_path (str): Path to audio file
            n_mfcc (int): Number of MFCC coefficients
            
        Returns:
            np.ndarray: Feature vector (71 dimensions)
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            
            # Calculate statistical features
            features = []
            features.extend(np.mean(mfcc, axis=1))  # Mean of each MFCC coefficient
            features.extend(np.std(mfcc, axis=1))   # Standard deviation
            features.extend(np.min(mfcc, axis=1))   # Minimum
            features.extend(np.max(mfcc, axis=1))   # Maximum
            features.extend(np.median(mfcc, axis=1)) # Median
            
            # Add additional features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            features.append(np.mean(spectral_centroids))
            features.append(np.std(spectral_centroids))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            features.append(np.mean(spectral_rolloff))
            features.append(np.std(spectral_rolloff))
            
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            features.append(np.mean(zero_crossing_rate))
            features.append(np.std(zero_crossing_rate))
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def extract_ui_features(self, audio_path):
        """Extract features for UI (39-dimensional embedding)
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            tuple: (normalized_vector, mfcc, delta_mfcc, delta2_mfcc, audio, sample_rate)
        """
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, n_fft=512,
                                       hop_length=160, n_mels=40, fmin=0, fmax=8000)
            d1 = librosa.feature.delta(mfcc)
            d2 = librosa.feature.delta(mfcc, order=2)
            vec = np.concatenate([mfcc.mean(1), d1.mean(1), d2.mean(1)])
            norm = np.linalg.norm(vec) + 1e-8
            return vec / norm, mfcc, d1, d2, y, sr
        except Exception as e:
            print(f"Feature extraction failed: {e}")
            raise
    
    def extract_features_batch(self, audio_paths):
        """Extract features from multiple audio files
        
        Args:
            audio_paths (list): List of audio file paths
            
        Returns:
            tuple: (features, labels) where failed extractions are filtered out
        """
        features = []
        valid_paths = []
        
        for audio_path in audio_paths:
            feature_vector = self.extract_mfcc_features(audio_path)
            if feature_vector is not None:
                features.append(feature_vector)
                valid_paths.append(audio_path)
        
        return np.array(features), valid_paths
    
    def extract_39dim_features(self, audio_path):
        """Extract 39-dimension MFCC features for Siamese model
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            np.ndarray: 39-dimensional feature vector
        """
        try:
            embedding, mfcc, d1, d2, y, sr = self.extract_ui_features(audio_path)
            return embedding
        except Exception as e:
            print(f"Error extracting 39-dim features from {audio_path}: {e}")
            return None

# Legacy function for backward compatibility
def extract_mfcc_features(audio_path, n_mfcc=13):
    """Legacy function for backward compatibility"""
    extractor = MFCCExtractor()
    return extractor.extract_mfcc_features(audio_path, n_mfcc)

# Cosine similarity function
def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors
    
    Args:
        a (np.ndarray): First vector
        b (np.ndarray): Second vector
        
    Returns:
        float: Cosine similarity score (0-1)
    """
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

# Advanced MFCC extraction for graph generation
def extract_mfcc_for_graphs(audio_path):
    """Extract MFCC features for advanced graph generation
    
    Args:
        audio_path (str): Path to audio file
        
    Returns:
        tuple: (embedding, mfcc, delta_mfcc, delta2_mfcc, audio, sample_rate)
    """
    extractor = MFCCExtractor()
    return extractor.extract_ui_features(audio_path)

# Training model MFCC extraction
def extract_mfcc_for_training(audio_path, n_mfcc=13):
    """Extract MFCC features for training models
    
    Args:
        audio_path (str): Path to audio file
        n_mfcc (int): Number of MFCC coefficients
        
    Returns:
        np.ndarray: 71-dimensional feature vector
    """
    extractor = MFCCExtractor()
    return extractor.extract_mfcc_features(audio_path, n_mfcc)

if __name__ == "__main__":
    # Test MFCC extraction
    extractor = MFCCExtractor()
    
    # Example usage
    test_audio = "uploads/speaker_9.wav"
    if Path(test_audio).exists():
        # Test 71-dim extraction
        features = extractor.extract_mfcc_features(test_audio)
        print(f"71-dim features shape: {features.shape}")
        
        # Test 39-dim extraction
        embedding = extractor.extract_39dim_features(test_audio)
        print(f"39-dim embedding shape: {embedding.shape}")
        
        # Test UI features
        vec, mfcc, d1, d2, y, sr = extractor.extract_ui_features(test_audio)
        print(f"UI features shape: {vec.shape}")
        print(f"MFCC shape: {mfcc.shape}")
        print(f"Delta shape: {d1.shape}")
        print(f"Delta2 shape: {d2.shape}")
    else:
        print(f"Test audio file not found: {test_audio}")
