#!/usr/bin/env python3
"""
Advanced Graph Generator for Speaker Recognition
Creates comprehensive MFCC and embedding visualizations based on reference images
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import librosa
from pathlib import Path
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde

# Set up matplotlib to avoid font issues
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']

def extract_mfcc_features(audio_path):
    """Extract MFCC features from audio file"""
    from extract_mfcc import extract_mfcc_for_graphs
    return extract_mfcc_for_graphs(audio_path)

def create_mfcc_comprehensive(mfcc, speaker_name, save_path):
    """Create comprehensive MFCC analysis like reference image"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor='white')
    axes = axes.flatten()
    
    # 1. Original MFCC Heatmap
    im1 = axes[0].imshow(mfcc, cmap='viridis', aspect='auto', interpolation='bilinear')
    axes[0].set_title('Original MFCC Coefficients', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Time Frames')
    axes[0].set_ylabel('MFCC Coefficients')
    plt.colorbar(im1, ax=axes[0])
    
    # 2. Delta MFCC
    delta_mfcc = np.gradient(mfcc, axis=1)
    im2 = axes[1].imshow(delta_mfcc, cmap='RdBu_r', aspect='auto')
    axes[1].set_title('Delta MFCC (Rate of Change)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Time Frames')
    axes[1].set_ylabel('MFCC Coefficients')
    plt.colorbar(im2, ax=axes[1])
    
    # 3. Energy Distribution
    energy = np.sum(mfcc**2, axis=0)
    axes[2].plot(energy, color='#3B82F6', linewidth=2)
    axes[2].fill_between(range(len(energy)), energy, alpha=0.3, color='#3B82F6')
    axes[2].set_title('MFCC Energy Distribution', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Time Frames')
    axes[2].set_ylabel('Energy')
    axes[2].grid(True, alpha=0.3)
    
    # 4. Coefficient Statistics
    stats_data = [np.mean(mfcc, axis=1), np.std(mfcc, axis=1), np.max(mfcc, axis=1), np.min(mfcc, axis=1)]
    labels = ['Mean', 'Std Dev', 'Max', 'Min']
    colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444']
    
    for i, (data, label, color) in enumerate(zip(stats_data, labels, colors)):
        axes[3].plot(range(13), data, label=label, color=color, linewidth=2, marker='o')
    axes[3].set_title('MFCC Coefficient Statistics', fontsize=12, fontweight='bold')
    axes[3].set_xlabel('MFCC Coefficient')
    axes[3].set_ylabel('Value')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    # 5. Spectral Rolloff Analysis
    spectral_rolloff = []
    for frame in range(mfcc.shape[1]):
        rolloff = np.sum(np.cumsum(mfcc[:, frame]) <= 0.85 * np.sum(mfcc[:, frame]))
        spectral_rolloff.append(rolloff)
    
    axes[4].plot(spectral_rolloff, color='#8B5CF6', linewidth=2)
    axes[4].fill_between(range(len(spectral_rolloff)), spectral_rolloff, alpha=0.3, color='#8B5CF6')
    axes[4].set_title('Spectral Rolloff Analysis', fontsize=12, fontweight='bold')
    axes[4].set_xlabel('Time Frames')
    axes[4].set_ylabel('Rolloff Coefficient')
    axes[4].grid(True, alpha=0.3)
    
    # 6. Coefficient Correlation Matrix
    corr_matrix = np.corrcoef(mfcc)
    im6 = axes[5].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    axes[5].set_title('MFCC Coefficient Correlation', fontsize=12, fontweight='bold')
    axes[5].set_xlabel('MFCC Coefficient')
    axes[5].set_ylabel('MFCC Coefficient')
    plt.colorbar(im6, ax=axes[5])
    
    plt.suptitle(f'Comprehensive MFCC Analysis - {speaker_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return save_path

def create_mfcc_individual(mfcc, speaker_name, save_path):
    """Create individual MFCC coefficient analysis"""
    fig, axes = plt.subplots(3, 5, figsize=(20, 12), facecolor='white')
    axes = axes.flatten()
    
    for i in range(13):
        # Plot individual coefficient
        axes[i].plot(mfcc[i, :], color=f'C{i}', linewidth=2)
        axes[i].fill_between(range(len(mfcc[i, :])), mfcc[i, :], alpha=0.3, color=f'C{i}')
        axes[i].set_title(f'MFCC Coefficient {i+1}', fontsize=10, fontweight='bold')
        axes[i].set_xlabel('Time Frames', fontsize=8)
        axes[i].set_ylabel('Amplitude', fontsize=8)
        axes[i].grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = np.mean(mfcc[i, :])
        std_val = np.std(mfcc[i, :])
        axes[i].text(0.02, 0.98, f'μ={mean_val:.2f}\nσ={std_val:.2f}', 
                    transform=axes[i].transAxes, fontsize=8,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Remove empty subplots
    for i in range(13, 15):
        fig.delaxes(axes[i])
    
    plt.suptitle(f'Individual MFCC Coefficient Analysis - {speaker_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return save_path

def create_mfcc_statistics(mfcc, speaker_name, save_path):
    """Create MFCC statistical analysis"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor='white')
    
    # 1. Distribution Analysis
    axes[0, 0].hist(mfcc.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('MFCC Value Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('MFCC Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Box Plot by Coefficient
    axes[0, 1].boxplot([mfcc[i, :] for i in range(13)], patch_artist=True)
    axes[0, 1].set_title('MFCC Coefficient Box Plot', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('MFCC Coefficient')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Variance Analysis
    variance = np.var(mfcc, axis=1)
    axes[0, 2].bar(range(1, 14), variance, color='green', alpha=0.7)
    axes[0, 2].set_title('MFCC Coefficient Variance', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('MFCC Coefficient')
    axes[0, 2].set_ylabel('Variance')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Mean Analysis
    mean_vals = np.mean(mfcc, axis=1)
    axes[1, 0].bar(range(1, 14), mean_vals, color='orange', alpha=0.7)
    axes[1, 0].set_title('MFCC Coefficient Mean', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('MFCC Coefficient')
    axes[1, 0].set_ylabel('Mean Value')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Skewness Analysis
    skewness = []
    for i in range(13):
        skew = np.mean(((mfcc[i, :] - np.mean(mfcc[i, :])) / np.std(mfcc[i, :]))**3)
        skewness.append(skew)
    
    axes[1, 1].bar(range(1, 14), skewness, color='red', alpha=0.7)
    axes[1, 1].set_title('MFCC Coefficient Skewness', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('MFCC Coefficient')
    axes[1, 1].set_ylabel('Skewness')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Kurtosis Analysis
    kurtosis = []
    for i in range(13):
        kurt = np.mean(((mfcc[i, :] - np.mean(mfcc[i, :])) / np.std(mfcc[i, :]))**4) - 3
        kurtosis.append(kurt)
    
    axes[1, 2].bar(range(1, 14), kurtosis, color='purple', alpha=0.7)
    axes[1, 2].set_title('MFCC Coefficient Kurtosis', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('MFCC Coefficient')
    axes[1, 2].set_ylabel('Kurtosis')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle(f'MFCC Statistical Analysis - {speaker_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return save_path

def create_similarity_heatmap(embedding, all_embeddings, speaker_name, save_path):
    """Create real similarity heatmap"""
    if all_embeddings is None or len(all_embeddings) == 0:
        # Create a simple self-similarity heatmap
        fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
        
        # Single speaker similarity matrix
        similarity_matrix = np.array([[1.0]])
        
        im = ax.imshow(similarity_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
        
        ax.set_xticks([0])
        ax.set_yticks([0])
        ax.set_xticklabels([speaker_name])
        ax.set_yticklabels([speaker_name])
        
        ax.text(0, 0, '1.000', ha="center", va="center", color="black", fontsize=12)
        
        ax.set_title('Speaker Similarity Heatmap', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Cosine Similarity')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return save_path
    
    # Combine current embedding with existing ones
    all_embs = [embedding] + all_embeddings
    names = [speaker_name] + [f"Speaker_{i+1}" for i in range(len(all_embeddings))]
    
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(all_embs)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
    
    im = ax.imshow(similarity_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45)
    ax.set_yticklabels(names)
    
    # Add text annotations
    for i in range(len(names)):
        for j in range(len(names)):
            text = ax.text(j, i, f'{similarity_matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title('Speaker Similarity Heatmap', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return save_path

def create_mfcc_advanced_analysis(mfcc, embedding, speaker_name, save_path):
    """Create advanced MFCC analysis"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor='white')
    axes = axes.flatten()
    
    # 1. MFCC Temporal Evolution
    temporal_evolution = []
    for i in range(13):
        evolution = np.convolve(mfcc[i, :], np.ones(10)/10, mode='same')
        temporal_evolution.append(evolution)
    
    for i, evolution in enumerate(temporal_evolution):
        axes[0].plot(evolution, label=f'MFCC {i+1}', alpha=0.7)
    axes[0].set_title('MFCC Temporal Evolution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Time Frames')
    axes[0].set_ylabel('Smoothed MFCC Value')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Spectral Centroid Evolution
    spectral_centroids = []
    for frame in range(mfcc.shape[1]):
        centroid = np.sum(np.arange(13) * mfcc[:, frame]) / np.sum(mfcc[:, frame])
        spectral_centroids.append(centroid)
    
    axes[1].plot(spectral_centroids, color='purple', linewidth=2)
    axes[1].fill_between(range(len(spectral_centroids)), spectral_centroids, alpha=0.3, color='purple')
    axes[1].set_title('Spectral Centroid Evolution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Time Frames')
    axes[1].set_ylabel('Spectral Centroid')
    axes[1].grid(True, alpha=0.3)
    
    # 3. MFCC Band Energy
    bands = [(0, 4), (5, 8), (9, 12)]
    band_names = ['Low', 'Mid', 'High']
    colors = ['blue', 'green', 'red']
    
    for (start, end), name, color in zip(bands, band_names, colors):
        band_energy = np.sum(mfcc[start:end+1, :]**2, axis=0)
        axes[2].plot(band_energy, label=f'{name} Band', color=color, linewidth=2)
    
    axes[2].set_title('MFCC Band Energy Analysis', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Time Frames')
    axes[2].set_ylabel('Band Energy')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # 4. Embedding Component Analysis
    components = ['MFCC Base', 'MFCC Delta', 'MFCC Delta-Delta']
    for i, (start, end, name) in enumerate([(0, 13, components[0]), (13, 26, components[1]), (26, 39, components[2])]):
        component_data = embedding[start:end]
        axes[3].bar(range(start, start+13), component_data, alpha=0.7, label=name)
    
    axes[3].set_title('Embedding Component Analysis', fontsize=12, fontweight='bold')
    axes[3].set_xlabel('Feature Index')
    axes[3].set_ylabel('Feature Value')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    # 5. Dynamic Range Analysis
    dynamic_ranges = []
    for i in range(13):
        drange = np.max(mfcc[i, :]) - np.min(mfcc[i, :])
        dynamic_ranges.append(drange)
    
    axes[4].bar(range(1, 14), dynamic_ranges, color='orange', alpha=0.7)
    axes[4].set_title('MFCC Dynamic Range Analysis', fontsize=12, fontweight='bold')
    axes[4].set_xlabel('MFCC Coefficient')
    axes[4].set_ylabel('Dynamic Range')
    axes[4].grid(True, alpha=0.3)
    
    # 6. Zero Crossing Rate
    zcr = []
    for i in range(13):
        zero_crossings = np.sum(np.diff(np.signbit(mfcc[i, :])))
        zcr.append(zero_crossings)
    
    axes[5].bar(range(1, 14), zcr, color='red', alpha=0.7)
    axes[5].set_title('MFCC Zero Crossing Rate', fontsize=12, fontweight='bold')
    axes[5].set_xlabel('MFCC Coefficient')
    axes[5].set_ylabel('Zero Crossings')
    axes[5].grid(True, alpha=0.3)
    
    plt.suptitle(f'Advanced MFCC Analysis - {speaker_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return save_path

def create_feature_importance(mfcc, embedding, speaker_name, save_path):
    """Create MFCC feature importance analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='white')
    
    # 1. Feature Correlation with Energy
    energy = np.sum(mfcc**2, axis=0)
    correlations = []
    for i in range(13):
        corr = np.corrcoef(mfcc[i, :], energy)[0, 1]
        correlations.append(corr)
    
    axes[0, 0].bar(range(1, 14), correlations, color='blue', alpha=0.7)
    axes[0, 0].set_title('MFCC-Energy Correlation', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('MFCC Coefficient')
    axes[0, 0].set_ylabel('Correlation with Energy')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Mutual Information (approximation)
    mutual_info = []
    for i in range(13):
        # Approximate mutual information using variance
        mi = np.var(mfcc[i, :]) / np.sum(np.var(mfcc, axis=1))
        mutual_info.append(mi)
    
    axes[0, 1].bar(range(1, 14), mutual_info, color='green', alpha=0.7)
    axes[0, 1].set_title('Feature Importance (Variance Ratio)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('MFCC Coefficient')
    axes[0, 1].set_ylabel('Importance Score')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Embedding Feature Magnitude
    base_magnitude = np.linalg.norm(embedding[:13])
    delta_magnitude = np.linalg.norm(embedding[13:26])
    delta2_magnitude = np.linalg.norm(embedding[26:39])
    
    magnitudes = [base_magnitude, delta_magnitude, delta2_magnitude]
    labels = ['MFCC Base', 'MFCC Delta', 'MFCC Delta-Delta']
    colors = ['blue', 'green', 'red']
    
    axes[1, 0].bar(labels, magnitudes, color=colors, alpha=0.7)
    axes[1, 0].set_title('Embedding Component Magnitude', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Magnitude')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Feature Stability Analysis
    stability = []
    for i in range(13):
        # Calculate stability as inverse of coefficient of variation
        cv = np.std(mfcc[i, :]) / np.abs(np.mean(mfcc[i, :]) + 1e-8)
        stability.append(1 / (1 + cv))
    
    axes[1, 1].bar(range(1, 14), stability, color='orange', alpha=0.7)
    axes[1, 1].set_title('Feature Stability Analysis', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('MFCC Coefficient')
    axes[1, 1].set_ylabel('Stability Score')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'MFCC Feature Importance Analysis - {speaker_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return save_path

def create_3d_density_plot(embedding, all_embeddings, speaker_name, save_path):
    """Create 3D density plot of embeddings"""
    fig = plt.figure(figsize=(12, 10), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate samples for density estimation
    np.random.seed(42)
    samples = [embedding + np.random.normal(0, 0.02, embedding.shape) for _ in range(100)]
    
    if all_embeddings:
        for emb in all_embeddings:
            samples.extend([emb + np.random.normal(0, 0.02, emb.shape) for _ in range(50)])
    
    all_points = np.array(samples)
    
    # Apply PCA for 3D visualization
    n_components = min(3, len(all_points), all_points.shape[1])
    pca = PCA(n_components=n_components)
    points_3d = pca.fit_transform(all_points)
    
    # Create density plot
    from scipy.stats import gaussian_kde
    
    # Calculate density
    try:
        xyz = points_3d.T
        kde = gaussian_kde(xyz)
        density = kde(xyz)
        
        # Plot with density coloring
        scatter = ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                          c=density, cmap='viridis', s=20, alpha=0.6)
        
        plt.colorbar(scatter, ax=ax, label='Density')
    except:
        # Fallback to simple scatter
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                  c='blue', s=20, alpha=0.6)
    
    # Highlight current speaker
    current_points = points_3d[:100]
    ax.scatter(current_points[:, 0], current_points[:, 1], current_points[:, 2], 
              c='red', s=50, alpha=0.9, marker='*', label=f'{speaker_name}')
    
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    ax.set_title('3D Embedding Density Plot', fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return save_path

def create_3d_pca_analysis(embedding, all_embeddings, speaker_name, save_path):
    """Create 3D PCA analysis of embeddings"""
    fig = plt.figure(figsize=(12, 10), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    
    # Prepare data
    all_embs = [embedding]
    labels = [speaker_name]
    colors = ['red']
    
    if all_embeddings:
        all_embs.extend(all_embeddings)
        labels.extend([f'Speaker_{i+1}' for i in range(len(all_embeddings))])
        colors.extend(['blue'] * len(all_embeddings))
    
    all_points = np.array(all_embs)
    
    # Generate synthetic data for visualization if we don't have enough samples
    if len(all_points) < 3:
        np.random.seed(42)
        synthetic_points = []
        for emb in all_points:
            # Generate multiple samples around each embedding
            for _ in range(10):
                synthetic_points.append(emb + np.random.normal(0, 0.01, emb.shape))
        all_points = np.array(synthetic_points)
        
        # Update labels and colors
        labels = []
        colors = []
        for i, emb in enumerate(all_embs):
            for j in range(10):
                labels.append(f'{speaker_name}_{j}' if i == 0 else f'Speaker_{i}_{j}')
                colors.append('red' if i == 0 else 'blue')
    
    # Apply PCA
    n_components = min(3, len(all_points), all_points.shape[1])
    pca = PCA(n_components=n_components)
    points_3d = pca.fit_transform(all_points)
    
    # Plot each speaker
    for i, (point, label, color) in enumerate(zip(points_3d, labels, colors)):
        ax.scatter(point[0], point[1], point[2], 
                  c=color, s=100, alpha=0.8, label=label, marker='o')
    
    # Add variance explained
    variance_explained = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({variance_explained[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({variance_explained[1]:.1%} variance)')
    ax.set_zlabel(f'PC3 ({variance_explained[2]:.1%} variance)')
    
    ax.set_title('3D PCA Analysis of Voice Embeddings', fontsize=14, fontweight='bold')
    ax.legend()
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return save_path

def create_3d_speaker_clustering(embedding, all_embeddings, speaker_name, save_path):
    """Create 3D speaker clustering visualization"""
    fig = plt.figure(figsize=(12, 10), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    
    # Prepare data
    all_embs = [embedding]
    if all_embeddings:
        all_embs.extend(all_embeddings)
    
    all_points = np.array(all_embs)
    
    # Generate synthetic data if we don't have enough samples
    if len(all_points) < 3:
        np.random.seed(42)
        synthetic_points = []
        for emb in all_embs:
            # Generate multiple samples around each embedding
            for _ in range(10):
                synthetic_points.append(emb + np.random.normal(0, 0.01, emb.shape))
        all_points = np.array(synthetic_points)
    
    # Apply t-SNE for better clustering visualization
    max_perplexity = min(30, len(all_points)-1)
    if max_perplexity < 1:
        max_perplexity = 1
    
    tsne = TSNE(n_components=3, random_state=42, perplexity=max_perplexity)
    points_3d = tsne.fit_transform(all_points)
    
    # Perform K-means clustering
    n_clusters = min(5, len(all_points))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(points_3d)
    
    # Plot clusters
    cluster_colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
    
    for i in range(n_clusters):
        cluster_points = points_3d[cluster_labels == i]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
                  c=[cluster_colors[i]], s=50, alpha=0.7, label=f'Cluster {i+1}')
    
    # Highlight current speaker
    ax.scatter(points_3d[0, 0], points_3d[0, 1], points_3d[0, 2],
              c='red', s=200, alpha=1.0, marker='*', 
              edgecolors='black', linewidth=2, label=f'{speaker_name}')
    
    # Draw cluster boundaries (convex hull approximation)
    from scipy.spatial import ConvexHull
    for i in range(n_clusters):
        cluster_points = points_3d[cluster_labels == i]
        if len(cluster_points) >= 4:
            try:
                hull = ConvexHull(cluster_points)
                for simplex in hull.simplices:
                    simplex_points = cluster_points[simplex]
                    ax.plot(simplex_points[:, 0], simplex_points[:, 1], simplex_points[:, 2],
                           'k-', alpha=0.3, linewidth=1)
            except:
                pass
    
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_zlabel('t-SNE Component 3')
    ax.set_title('3D Speaker Clustering Analysis', fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return save_path

def create_all_advanced_graphs(audio_path, speaker_name, output_dir, all_embeddings=None):
    """Create all advanced graphs for a speaker"""
    
    # Create output directory
    output_path = Path(output_dir) / speaker_name
    output_path.mkdir(exist_ok=True)
    
    # Extract features
    embedding, mfcc, d1, d2, y, sr = extract_mfcc_features(audio_path)
    
    if embedding is None:
        return None
    
    # Create all graphs
    graphs = {}
    
    # MFCC Analysis Graphs
    graphs['comprehensive'] = create_mfcc_comprehensive(mfcc, speaker_name, 
                                                      output_path / f'{speaker_name}_mfcc_comprehensive.png')
    graphs['individual'] = create_mfcc_individual(mfcc, speaker_name, 
                                                output_path / f'{speaker_name}_mfcc_individual.png')
    graphs['statistics'] = create_mfcc_statistics(mfcc, speaker_name, 
                                                 output_path / f'{speaker_name}_mfcc_statistics.png')
    graphs['similarity_heatmap'] = create_similarity_heatmap(embedding, all_embeddings, speaker_name,
                                                          output_path / f'{speaker_name}_real_similarity_heatmap.png')
    graphs['advanced_analysis'] = create_mfcc_advanced_analysis(mfcc, embedding, speaker_name,
                                                             output_path / f'{speaker_name}_mfcc_advanced_analysis.png')
    graphs['feature_importance'] = create_feature_importance(mfcc, embedding, speaker_name,
                                                            output_path / f'{speaker_name}_mfcc_advanced_feature_importance.png')
    
    # 3D Embedding Graphs
    graphs['3d_density'] = create_3d_density_plot(embedding, all_embeddings, speaker_name,
                                                  output_path / f'{speaker_name}_3d_density_plot.png')
    graphs['3d_pca'] = create_3d_pca_analysis(embedding, all_embeddings, speaker_name,
                                               output_path / f'{speaker_name}_3d_pca_analysis.png')
    graphs['3d_clustering'] = create_3d_speaker_clustering(embedding, all_embeddings, speaker_name,
                                                         output_path / f'{speaker_name}_3d_speaker_clustering.png')
    
    return graphs

if __name__ == "__main__":
    import time
    
    # Example usage
    audio_file = "test_samples/Speaker_0001_00054.wav"
    speaker_name = "test_speaker"
    output_dir = "mfcc_output"
    
    graphs = create_all_advanced_graphs(audio_file, speaker_name, output_dir)
    
    if graphs:
        print("Advanced graphs created successfully:")
        for graph_type, path in graphs.items():
            print(f"  {graph_type}: {path}")
    else:
        print("Failed to create graphs")
