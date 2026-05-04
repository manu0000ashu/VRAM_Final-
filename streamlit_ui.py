#!/usr/bin/env python3
"""
Comprehensive Streamlit UI for Audio Database Manager
Matches all backend functionality: upload, compare, database listing, and graph generation
"""

import streamlit as st
import os
import json
import shutil
import pickle
import tempfile
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from datetime import datetime
from extract_mfcc import MFCCExtractor
from advanced_graph_generator import create_all_advanced_graphs

# Set page config
st.set_page_config(
    page_title="Audio Database Manager",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
/* Tab styling - Always visible and larger */
.stTabs [data-baseweb="tab-list"] {
    background-color: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 0.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.stTabs [data-baseweb="tab"] {
    background-color: white;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    font-size: 1.1rem;
    font-weight: 600;
    font-family: "Inter", sans-serif;
    color: #1E293B;
    transition: all 0.2s ease;
    min-height: 60px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.stTabs [data-baseweb="tab"]:hover {
    background-color: #F1F5F9;
    border-color: #CBD5E1;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}

.stTabs [aria-selected="true"] {
    background-color: #3B82F6 !important;
    color: white !important;
    border-color: #3B82F6 !important;
    box-shadow: 0 4px 16px rgba(59,130,246,0.3);
}

.stTabs [aria-selected="true"]:hover {
    background-color: #2563EB !important;
    border-color: #2563EB !important;
}

.metric-card {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    margin-bottom: 1rem;
}

.success-box {
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    border-radius: 8px;
    padding: 1rem;
    color: #166534;
}

.error-box {
    background: #FEF2F2;
    border: 1px solid #FECACA;
    border-radius: 8px;
    padding: 1rem;
    color: #991B1B;
}

.warning-box {
    background: #FFFBEB;
    border: 1px solid #FED7AA;
    border-radius: 8px;
    padding: 1rem;
    color: #92400E;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'db_initialized' not in st.session_state:
    st.session_state.db_initialized = False
    st.session_state.embeddings = {}
    st.session_state.metadata = {}
    st.session_state.encoder = None
    st.session_state.scaler = None
    st.session_state.siamese_embeddings = {}
    st.session_state.extractor = None

def initialize_database():
    """Initialize the audio database"""
    try:
        # Database paths
        db_path = Path("audio_database")
        db_path.mkdir(exist_ok=True)
        
        audio_dir = db_path / "audio_files"
        audio_dir.mkdir(exist_ok=True)
        
        # Output directories
        all_output_dir = Path("all_output")
        all_output_dir.mkdir(exist_ok=True)
        option_1_folder = all_output_dir / "option_1_folder"
        option_1_folder.mkdir(exist_ok=True)
        option_2_folder = all_output_dir / "option_2_folder"
        option_2_folder.mkdir(exist_ok=True)
        
        # Database files
        embeddings_file = db_path / "embeddings.json"
        metadata_file = db_path / "metadata.json"
        
        # Load existing data
        if embeddings_file.exists():
            with open(embeddings_file, 'r') as f:
                data = json.load(f)
                st.session_state.embeddings = {name: np.array(embedding) for name, embedding in data.items()}
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                st.session_state.metadata = json.load(f)
        
        # Initialize feature extractor
        st.session_state.extractor = MFCCExtractor()
        
        # Load Siamese model
        load_siamese_model()
        
        st.session_state.db_initialized = True
        return True
        
    except Exception as e:
        st.error(f"Error initializing database: {e}")
        return False

def load_siamese_model():
    """Load the Siamese model"""
    try:
        model_dir = Path("siamese_model_39dim")
        
        if not model_dir.exists():
            return False
        
        # Load encoder
        with open(model_dir / "encoder.pkl", 'rb') as f:
            st.session_state.encoder = pickle.load(f)
        
        # Load scaler
        with open(model_dir / "scaler.pkl", 'rb') as f:
            st.session_state.scaler = pickle.load(f)
        
        # Load speaker embeddings
        with open(model_dir / "speaker_embeddings_39dim.json", 'r') as f:
            embeddings_data = json.load(f)
        
        for speaker_name, embedding_list in embeddings_data.items():
            st.session_state.siamese_embeddings[speaker_name] = np.array(embedding_list)
        
        return True
        
    except Exception as e:
        return False

def save_embeddings():
    """Save embeddings to file"""
    try:
        embeddings_file = Path("audio_database/embeddings.json")
        data = {name: embedding.tolist() for name, embedding in st.session_state.embeddings.items()}
        with open(embeddings_file, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        st.error(f"Error saving embeddings: {e}")

def save_metadata():
    """Save metadata to file"""
    try:
        metadata_file = Path("audio_database/metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(st.session_state.metadata, f, indent=2)
    except Exception as e:
        st.error(f"Error saving metadata: {e}")

def compare_audio_features(embedding, stored_embedding):
    """Compare two audio embeddings using multiple metrics"""
    # 1. Cosine similarity
    cosine_sim = np.dot(embedding, stored_embedding) / (
        np.linalg.norm(embedding) * np.linalg.norm(stored_embedding)
    )
    
    # 2. Euclidean distance
    euclidean_dist = np.linalg.norm(embedding - stored_embedding)
    
    # 3. Manhattan distance
    manhattan_dist = np.sum(np.abs(embedding - stored_embedding))
    
    # 4. Siamese model comparison
    siamese_sim = 0.0
    if st.session_state.encoder is not None and st.session_state.scaler is not None:
        try:
            diff = np.abs(embedding - stored_embedding)
            diff_scaled = st.session_state.scaler.transform([diff])
            similarity_prob = st.session_state.encoder.predict_proba(diff_scaled)[0]
            siamese_sim = similarity_prob[1]
        except:
            siamese_sim = 0.0
    
    # 5. Combined score
    norm_euclidean = max(0, 1 - (euclidean_dist / 50))
    norm_manhattan = max(0, 1 - (manhattan_dist / 500))
    
    combined_score = (
        0.35 * cosine_sim + 
        0.25 * norm_euclidean + 
        0.10 * norm_manhattan +
        0.30 * siamese_sim
    )
    
    return {
        'cosine': cosine_sim,
        'euclidean': euclidean_dist,
        'manhattan': manhattan_dist,
        'siamese': siamese_sim,
        'combined': combined_score
    }

# Initialize database
if not st.session_state.db_initialized:
    with st.spinner("Initializing database..."):
        if initialize_database():
            st.success("✅ Database initialized successfully!")
        else:
            st.error("❌ Failed to initialize database")
            st.stop()

# Header
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 style="font-size: 3rem; font-weight: 700; margin-bottom: 1rem;">
        🎵 Audio Database Manager
    </h1>
    <p style="font-size: 1.2rem; color: #64748b; margin-bottom: 2rem;">
        Upload, Compare, and Manage Speaker Audio Files with Advanced Graph Generation
    </p>
</div>
""", unsafe_allow_html=True)

# Database status in sidebar
st.sidebar.markdown("### 📊 Database Status")
st.sidebar.metric("Total Speakers", len(st.session_state.embeddings))
st.sidebar.metric("Siamese Model", "✅ Loaded" if st.session_state.encoder else "❌ Not Available")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎵 Upload Audio", "🔍 Compare Audio", "📋 Database", "📈 Graphs"])

# Tab 1: Upload Audio
with tab1:
    st.markdown("## 🎵 Upload New Audio")
    st.write("Upload audio files or MFCC features to add to the database")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File type selection
        file_type = st.radio("File Type", ["Audio File", "MFCC File"], help="Choose whether to upload a raw audio file or pre-extracted MFCC features")
        
        if file_type == "Audio File":
            uploaded_file = st.file_uploader(
                "Upload Audio File",
                type=["wav", "mp3", "flac", "m4a"],
                help="Upload audio file for feature extraction"
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload MFCC File",
                type=["npy", "json", "pkl"],
                help="Upload pre-extracted MFCC features"
            )
    
    with col2:
        speaker_name = st.text_input(
            "Speaker Name",
            placeholder="e.g., speaker_30",
            help="Enter a unique name for this speaker"
        )
    
    # Upload button
    if st.button("🚀 Upload to Database", type="primary", disabled=not uploaded_file or not speaker_name):
        with st.spinner("Processing audio..."):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Process based on file type
                use_mfcc = (file_type == "MFCC File")
                
                # Extract features
                embedding, mfcc, delta, delta2, y, sr = st.session_state.extractor.extract_or_load_features(tmp_path, use_mfcc)
                
                # Generate unique filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_ext = Path(uploaded_file.name).suffix
                audio_filename = f"{speaker_name}_{timestamp}{file_ext}"
                
                if use_mfcc:
                    audio_save_path = None
                    duration = 0.0
                else:
                    # Copy to audio database
                    audio_save_path = Path("audio_database/audio_files") / audio_filename
                    shutil.copy2(tmp_path, audio_save_path)
                    duration = librosa.get_duration(filename=tmp_path)
                
                # Store in database
                st.session_state.embeddings[speaker_name] = embedding
                st.session_state.metadata[speaker_name] = {
                    "audio_file": audio_save_path.name if audio_save_path else "MFCC_FILE",
                    "upload_time": datetime.now().isoformat(),
                    "feature_shape": list(embedding.shape),
                    "sample_rate": sr,
                    "duration": duration,
                    "file_type": "MFCC" if use_mfcc else "AUDIO"
                }
                
                # Save database
                save_embeddings()
                save_metadata()
                
                # Clean up
                Path(tmp_path).unlink()
                
                # Success message
                st.success(f"✅ {file_type} uploaded successfully!")
                st.info(f"🎯 Speaker: {speaker_name}")
                st.info(f"📊 Features: {embedding.shape}")
                if not use_mfcc:
                    st.info(f"⏱️ Duration: {duration:.2f} seconds")
                
                # Generate graphs
                with st.spinner("Generating graphs..."):
                    try:
                        all_embeddings_list = [v for k, v in st.session_state.embeddings.items() if k != speaker_name]
                        graphs = create_all_advanced_graphs(
                            audio_path=str(audio_save_path) if audio_save_path else tmp_path,
                            speaker_name=speaker_name,
                            output_dir="all_output/option_1_folder",
                            all_embeddings=all_embeddings_list
                        )
                        if graphs:
                            st.success("✅ Graphs generated successfully!")
                            for graph_type, path in graphs.items():
                                st.write(f"📊 {graph_type}: {path}")
                    except Exception as e:
                        st.warning(f"⚠️ Could not generate graphs: {e}")
                
                # Refresh
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error processing audio: {e}")
                # Clean up on error
                try:
                    Path(tmp_path).unlink()
                except:
                    pass

# Tab 2: Compare Audio
with tab2:
    st.markdown("## 🔍 Compare Audio")
    st.write("Compare audio files against the database")
    
    if not st.session_state.embeddings:
        st.warning("⚠️ No audio files in database! Upload some audio first.")
    else:
        # Show available speakers
        st.subheader("📋 Available Speakers in Database:")
        speakers_list = list(st.session_state.embeddings.keys())
        for i, speaker in enumerate(speakers_list, 1):
            duration = st.session_state.metadata[speaker]["duration"]
            st.write(f"  {i}. {speaker} ({duration:.2f}s)")
        
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # File type selection for comparison
            compare_file_type = st.radio("Comparison File Type", ["Audio File", "MFCC File"])
            
            if compare_file_type == "Audio File":
                compare_file = st.file_uploader(
                    "Upload Audio File to Compare",
                    type=["wav", "mp3", "flac", "m4a"]
                )
            else:
                compare_file = st.file_uploader(
                    "Upload MFCC File to Compare",
                    type=["npy", "json", "pkl"]
                )
        
        with col2:
            compare_name = st.text_input(
                "Comparison Name (Optional)",
                placeholder="e.g., Test_Speaker_30",
                help="Name for saving comparison graphs"
            )
        
        # Compare button
        if st.button("🔍 Compare Audio", type="primary", disabled=not compare_file):
            with st.spinner("Comparing audio..."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{compare_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(compare_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Process based on file type
                    use_mfcc = (compare_file_type == "MFCC File")
                    
                    # Extract features
                    embedding, mfcc, delta, delta2, y, sr = st.session_state.extractor.extract_or_load_features(tmp_path, use_mfcc)
                    
                    # Compare with all database entries
                    results = {}
                    for speaker_name, stored_embedding in st.session_state.embeddings.items():
                        metrics = compare_audio_features(embedding, stored_embedding)
                        results[speaker_name] = metrics
                    
                    # Sort by combined score
                    sorted_speakers = sorted(results.items(), key=lambda x: x[1]['combined'], reverse=True)
                    
                    # Display results
                    st.subheader("🎯 Comparison Results")
                    
                    # Results table
                    df_data = []
                    for i, (speaker, metrics) in enumerate(sorted_speakers, 1):
                        is_match = (metrics['cosine'] > 0.998 and 
                                   metrics['euclidean'] < 0.5 and 
                                   metrics['siamese'] > 0.99 and 
                                   metrics['combined'] > 0.95)
                        
                        df_data.append({
                            'Rank': i,
                            'Speaker': speaker,
                            'Cosine': f"{metrics['cosine']:.4f}",
                            'Euclidean': f"{metrics['euclidean']:.2f}",
                            'Siamese': f"{metrics['siamese']:.4f}" if metrics['siamese'] > 0 else "N/A",
                            'Combined': f"{metrics['combined']:.4f}",
                            'Status': "✅ MATCH" if is_match else "❌ NO MATCH"
                        })
                    
                    df = pd.DataFrame(df_data)
                    st.dataframe(df, width='stretch')
                    
                    # Analysis section
                    best_match = sorted_speakers[0]
                    second_best = sorted_speakers[1] if len(sorted_speakers) > 1 else (None, {'combined': 0})
                    gap = best_match[1]['combined'] - second_best[1]['combined']
                    
                    st.subheader("📊 Analysis")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Best Match", best_match[0])
                        st.metric("Combined Score", f"{best_match[1]['combined']:.4f}")
                    
                    with col2:
                        st.metric("Gap", f"{gap:.4f}")
                        st.metric("Cosine Similarity", f"{best_match[1]['cosine']:.4f}")
                    
                    with col3:
                        st.metric("Euclidean Distance", f"{best_match[1]['euclidean']:.4f}")
                        if best_match[1]['siamese'] > 0:
                            st.metric("Siamese Model", f"{best_match[1]['siamese']:.4f}")
                    
                    # Determine match status
                    is_clear_winner = gap > 0.15
                    is_high_similarity = best_match[1]['cosine'] > 0.998
                    is_low_distance = best_match[1]['euclidean'] < 0.5
                    is_high_siamese = best_match[1]['siamese'] > 0.99 if best_match[1]['siamese'] > 0 else False
                    
                    if is_high_similarity and is_low_distance and is_high_siamese and (is_clear_winner or best_match[1]['siamese'] >= 0.999):
                        st.success(f"🏆 CONFIDENT MATCH: {best_match[0]}")
                        st.info("This speaker is VERIFIED in the database!")
                    elif best_match[1]['siamese'] > 0.5 and best_match[1]['siamese'] < 0.99:
                        st.warning(f"❓ BORDERLINE MATCH: {best_match[0]}")
                        st.info("This is likely an UNKNOWN SPEAKER or needs verification.")
                    else:
                        st.error(f"🚫 UNKNOWN SPEAKER")
                        st.info("This speaker is NOT in the database!")
                    
                    # Generate comparison graphs
                    if compare_name or st.button("Generate Comparison Graphs"):
                        if not compare_name:
                            compare_name = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        
                        with st.spinner("Generating comparison graphs..."):
                            try:
                                all_embeddings_list = list(st.session_state.embeddings.values())
                                graphs = create_all_advanced_graphs(
                                    audio_path=tmp_path,
                                    speaker_name=compare_name,
                                    output_dir="all_output/option_2_folder",
                                    all_embeddings=all_embeddings_list
                                )
                                if graphs:
                                    st.success("✅ Comparison graphs generated successfully!")
                                    for graph_type, path in graphs.items():
                                        st.write(f"📊 {graph_type}: {path}")
                                else:
                                    st.warning("⚠️ Could not generate graphs")
                            except Exception as e:
                                st.warning(f"⚠️ Error generating graphs: {e}")
                    
                    # Clean up
                    Path(tmp_path).unlink()
                    
                except Exception as e:
                    st.error(f"❌ Error comparing audio: {e}")
                    try:
                        Path(tmp_path).unlink()
                    except:
                        pass

# Tab 3: Database
with tab3:
    st.markdown("## 📋 Database Contents")
    st.write("View all speakers and audio files in the database")
    
    if not st.session_state.embeddings:
        st.warning("📭 Database is empty!")
    else:
        st.info(f"📊 Total entries: {len(st.session_state.embeddings)}")
        
        # Create dataframe for database contents
        db_data = []
        for speaker_name in st.session_state.embeddings.keys():
            metadata = st.session_state.metadata[speaker_name]
            db_data.append({
                'Speaker': speaker_name,
                'File': metadata['audio_file'],
                'Duration': f"{metadata['duration']:.2f}s",
                'Upload Time': metadata['upload_time'][:19],
                'Features': str(metadata['feature_shape']),
                'Type': metadata.get('file_type', 'AUDIO')  # Fallback to 'AUDIO' if file_type not present
            })
        
        df = pd.DataFrame(db_data)
        st.dataframe(df, width='stretch')
        
        # Audio file links
        st.markdown("---")
        st.subheader("🎵 Audio Files")
        st.write("Access audio files from the database:")
        
        audio_dir = Path("audio_database/audio_files")
        if audio_dir.exists():
            audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3"))
            if audio_files:
                for audio_file in audio_files:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"📁 {audio_file.name}")
                    with col2:
                        # Provide download link
                        with open(audio_file, "rb") as file:
                            st.download_button(
                                label="Download",
                                data=file.read(),
                                file_name=audio_file.name,
                                mime="audio/wav"
                            )
            else:
                st.info("No audio files found")
        else:
            st.warning("Audio directory not found")

# Tab 4: Graphs
with tab4:
    st.markdown("## 📈 Generated Graphs")
    st.write("View and download generated analysis graphs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Option 1 Folder (Uploaded Audio)")
        option1_dir = Path("all_output/option_1_folder")
        if option1_dir.exists():
            speaker_folders = [d for d in option1_dir.iterdir() if d.is_dir()]
            if speaker_folders:
                selected_speaker = st.selectbox("Select Speaker", speaker_folders, key="opt1_speaker")
                if selected_speaker:
                    graph_files = list(selected_speaker.glob("*.png"))
                    if graph_files:
                        for graph_file in graph_files:
                            st.image(str(graph_file), caption=graph_file.name, width='stretch')
                            with open(graph_file, "rb") as file:
                                st.download_button(
                                    label=f"Download {graph_file.name}",
                                    data=file.read(),
                                    file_name=graph_file.name,
                                    mime="image/png"
                                )
                    else:
                        st.info("No graphs found for this speaker")
            else:
                st.info("No speaker folders found")
        else:
            st.warning("Option 1 folder not found")
    
    with col2:
        st.subheader("📊 Option 2 Folder (Comparisons)")
        option2_dir = Path("all_output/option_2_folder")
        if option2_dir.exists():
            comparison_folders = [d for d in option2_dir.iterdir() if d.is_dir()]
            if comparison_folders:
                selected_comparison = st.selectbox("Select Comparison", comparison_folders, key="opt2_comparison")
                if selected_comparison:
                    graph_files = list(selected_comparison.glob("*.png"))
                    if graph_files:
                        for graph_file in graph_files:
                            st.image(str(graph_file), caption=graph_file.name, width='stretch')
                            with open(graph_file, "rb") as file:
                                st.download_button(
                                    label=f"Download {graph_file.name}",
                                    data=file.read(),
                                    file_name=graph_file.name,
                                    mime="image/png"
                                )
                    else:
                        st.info("No graphs found for this comparison")
            else:
                st.info("No comparison folders found")
        else:
            st.warning("Option 2 folder not found")

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #64748b; border-top: 1px solid #E2E8F0; margin-top: 2rem;">
    <p>🎵 Audio Database Manager - Speaker Recognition System</p>
    <p style="font-size: 0.8rem;">39-Dimension MFCC Features with Siamese Network</p>
</div>
""", unsafe_allow_html=True)
