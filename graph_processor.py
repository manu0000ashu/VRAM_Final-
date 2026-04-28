#!/usr/bin/env python3
"""
Graph Processor Script
Generates comprehensive graphs and 3D dimensional visualizations for uploaded speakers
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
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
    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512,
                                   hop_length=160, n_mels=40, fmin=0, fmax=8000)
        d1 = librosa.feature.delta(mfcc)
        d2 = librosa.feature.delta(mfcc, order=2)
        
        # Create 39-dimensional embedding
        vec = np.concatenate([mfcc.mean(1), d1.mean(1), d2.mean(1)])
        norm = np.linalg.norm(vec) + 1e-8
        embedding = vec / norm
        
        return embedding, mfcc, d1, d2, y, sr
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None, None, None, None, None, None

def create_mfcc_comprehensive(mfcc, speaker_name, save_path):
    """Create comprehensive MFCC analysis"""
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
    axes[1].set_ylabel('Delta Coefficients')
    plt.colorbar(im2, ax=axes[1])
    
    # 3. Energy Distribution
    energy = np.sum(mfcc**2, axis=0)
    axes[2].plot(energy, color='#3B82F6', linewidth=2)
    axes[2].fill_between(range(len(energy)), energy, alpha=0.3, color='#3B82F6')
    axes[2].set_title('MFCC Energy Distribution', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Time Frames')
    axes[2].set_ylabel('Energy')
    
    # 4. Coefficient Variance
    variance = np.var(mfcc, axis=1)
    coeff_indices = range(1, 14)
    axes[3].bar(coeff_indices, variance, color='#10B981', alpha=0.8)
    axes[3].set_title('MFCC Coefficient Variance', fontsize=12, fontweight='bold')
    axes[3].set_xlabel('MFCC Coefficient Index')
    axes[3].set_ylabel('Variance')
    
    # 5. Spectral Centroid
    spectral_centroids = []
    for frame in range(mfcc.shape[1]):
        centroid = np.sum(np.arange(13) * mfcc[:, frame]) / np.sum(mfcc[:, frame])
        spectral_centroids.append(centroid)
    
    axes[4].plot(spectral_centroids, color='#F59E0B', linewidth=2)
    axes[4].fill_between(range(len(spectral_centroids)), spectral_centroids, alpha=0.3, color='#F59E0B')
    axes[4].set_title('Spectral Centroid Evolution', fontsize=12, fontweight='bold')
    axes[4].set_xlabel('Time Frames')
    axes[4].set_ylabel('Spectral Centroid')
    
    # 6. Correlation Matrix
    correlation_matrix = np.corrcoef(mfcc)
    im6 = axes[5].imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
    axes[5].set_title('MFCC Coefficient Correlation', fontsize=12, fontweight='bold')
    axes[5].set_xlabel('MFCC Coefficient')
    axes[5].set_ylabel('MFCC Coefficient')
    plt.colorbar(im6, ax=axes[5])
    
    plt.suptitle(f'MFCC Comprehensive Analysis - {speaker_name}', fontsize=16, fontweight='bold')
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
    try:
        xyz = points_3d.T
        kde = gaussian_kde(xyz)
        density = kde(xyz)
        
        # Plot points colored by density
        scatter = ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                           c=density, cmap='viridis', s=50, alpha=0.8)
        
        plt.colorbar(scatter, ax=ax, label='Density')
        
    except Exception as e:
        # Fallback to simple scatter plot
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                  c='#3B82F6', s=50, alpha=0.8)
    
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    ax.set_title(f'3D Voice Embedding Density - {speaker_name}', fontsize=14, fontweight='bold')
    
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
    
    # Generate synthetic data if we don't have enough samples
    if len(all_points) < 3:
        np.random.seed(42)
        synthetic_points = []
        for emb in all_embs:
            for _ in range(10):
                synthetic_points.append(emb + np.random.normal(0, 0.01, emb.shape))
        all_points = np.array(synthetic_points)
        
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
    ax.set_title(f'3D PCA Analysis - {speaker_name}', fontsize=14, fontweight='bold')
    ax.legend()
    
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
    
    for i, (point, label) in enumerate(zip(points_3d, cluster_labels)):
        ax.scatter(point[0], point[1], point[2], 
                  c=[cluster_colors[label]], s=100, alpha=0.8, marker='o')
    
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_zlabel('t-SNE Component 3')
    ax.set_title(f'3D Speaker Clustering - {speaker_name}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return save_path

def process_speaker_graphs(audio_path, speaker_name, output_dir="mfcc_output"):
    """Main function to process speaker and generate all graphs"""
    print(f"🚀 Processing speaker: {speaker_name}")
    print(f"📁 Audio file: {audio_path}")
    print(f"📂 Output directory: {output_dir}")
    
    # Extract features
    print("🔍 Extracting MFCC features...")
    embedding, mfcc, d1, d2, y, sr = extract_mfcc_features(audio_path)
    
    if embedding is None:
        print("❌ Failed to extract features")
        return None
    
    print(f"✅ Features extracted - Embedding shape: {embedding.shape}")
    
    # Create output directory
    safe_name = speaker_name.replace(' ', '_').replace('-', '_')
    speaker_dir = Path(output_dir) / safe_name
    speaker_dir.mkdir(exist_ok=True)
    
    # Generate all graphs
    print("📊 Generating comprehensive graphs...")
    
    graphs = {}
    
    try:
        # MFCC Comprehensive Analysis
        graphs['mfcc_comprehensive'] = create_mfcc_comprehensive(
            mfcc, speaker_name, speaker_dir / f'{safe_name}_mfcc_comprehensive.png'
        )
        
        # 3D Density Plot
        graphs['3d_density'] = create_3d_density_plot(
            embedding, [], speaker_name, speaker_dir / f'{safe_name}_3d_density_plot.png'
        )
        
        # 3D PCA Analysis
        graphs['3d_pca'] = create_3d_pca_analysis(
            embedding, [], speaker_name, speaker_dir / f'{safe_name}_3d_pca_analysis.png'
        )
        
        # 3D Speaker Clustering
        graphs['3d_clustering'] = create_3d_speaker_clustering(
            embedding, [], speaker_name, speaker_dir / f'{safe_name}_3d_speaker_clustering.png'
        )
        
        print(f"✅ Generated {len(graphs)} graphs:")
        for graph_type, path in graphs.items():
            print(f"   {graph_type}: {path}")
        
        # Save metadata
        metadata = {
            'speaker': speaker_name,
            'audio_file': Path(audio_path).name,
            'processed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'embedding': embedding.tolist(),
            'dimensions': len(embedding),
            'graphs_created': list(graphs.keys()),
            'graphs_count': len(graphs),
            'output_directory': str(speaker_dir)
        }
        
        metadata_path = speaker_dir / f'{safe_name}_processing_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📝 Metadata saved to: {metadata_path}")
        
        return graphs
        
    except Exception as e:
        print(f"❌ Error generating graphs: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function for script execution"""
    if len(sys.argv) < 3:
        print("Usage: python graph_processor.py <audio_path> <speaker_name> [output_dir]")
        print("Example: python graph_processor.py uploads/ASHIRWAD.wav ASHIRWAD mfcc_output")
        return
    
    audio_path = sys.argv[1]
    speaker_name = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "mfcc_output"
    
    # Check if audio file exists
    if not Path(audio_path).exists():
        print(f"❌ Audio file not found: {audio_path}")
        return
    
    # Process speaker graphs
    graphs = process_speaker_graphs(audio_path, speaker_name, output_dir)
    
    if graphs:
        print(f"\n🎉 Successfully processed {speaker_name}!")
        print(f"📁 All graphs saved to: {Path(output_dir) / speaker_name.replace(' ', '_')}")
    else:
        print(f"\n❌ Failed to process {speaker_name}")

if __name__ == "__main__":
    main()
