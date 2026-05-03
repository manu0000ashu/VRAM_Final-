#!/usr/bin/env python3
"""
Streamlit UI for Speaker Recognition System
Uses the working Siamese model for speaker identification
"""

import streamlit as st
import sys
import json
import tempfile
import time
import numpy as np
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Speaker Recognition System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

:root {
    --bg: #0A0A0F;
    --surface: #1A1A2E;
    --surface2: #16213E;
    --accent: #7C3AED;
    --accent2: #A78BFA;
    --text: #FFFFFF;
    --muted: #9CA3AF;
    --border: #374151;
    --glow: rgba(124, 58, 237, 0.3);
}

.main {
    background: var(--bg);
    color: var(--text);
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background: var(--bg);
}

h1, h2, h3 {
    color: var(--text);
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, #5B21B6 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.875rem 2rem;
    font-weight: 600;
    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(124, 58, 237, 0.55);
}

.stFileUploader > div {
    background: var(--surface2);
    border: 2px dashed var(--border);
    border-radius: 12px;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--surface);
    border-radius: 12px;
    padding: 4px;
}

.stTabs [aria-selected="true"] {
    background: var(--accent);
    color: white;
}

.siamese-badge {
    background: linear-gradient(135deg, #10B981 0%, #059669 100%);
    color: white;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.875rem;
    font-weight: 600;
    margin-left: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.speaker_embeddings = {}
    st.session_state.encoder = None
    st.session_state.scaler = None

def load_siamese_model():
    """Load the Siamese model"""
    try:
        import pickle
        import numpy as np
        from extract_mfcc import MFCCExtractor
        
        model_dir = Path("siamese_model_39dim")
        
        if not model_dir.exists():
            st.error("❌ Siamese model not found!")
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
            st.session_state.speaker_embeddings[speaker_name] = np.array(embedding_list)
        
        st.session_state.feature_extractor = MFCCExtractor()
        st.session_state.model_loaded = True
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return False

def identify_speaker(audio_path, threshold=0.5):
    """Identify speaker using Siamese model"""
    
    if not st.session_state.model_loaded:
        return {"error": "Model not loaded"}
    
    try:
        # Extract 39-dim features
        embedding, mfcc, d1, d2, y, sr = st.session_state.feature_extractor.extract_ui_features(audio_path)
        
        # Calculate distances to all speaker embeddings
        distances = {}
        best_speaker = None
        best_distance = float('inf')
        
        for speaker_name, speaker_embedding in st.session_state.speaker_embeddings.items():
            # Calculate feature difference
            diff = np.abs(embedding - speaker_embedding)
            
            # Scale the difference
            diff_scaled = st.session_state.scaler.transform([diff])
            
            # Predict similarity
            similarity_prob = st.session_state.encoder.predict_proba(diff_scaled)[0]
            similarity_score = similarity_prob[1]
            distance = 1 - similarity_score
            
            distances[speaker_name] = distance
            
            if distance < best_distance:
                best_distance = distance
                best_speaker = speaker_name
        
        return {
            'speaker': best_speaker,
            'distance': best_distance,
            'is_match': best_distance < threshold,
            'confidence': 1 - best_distance,
            'all_distances': distances,
            'features_used': 39
        }
        
    except Exception as e:
        return {"error": f"Identification failed: {e}"}

# Header
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 style="font-size: 3rem; font-weight: 700; margin-bottom: 1rem;">
        🧠 Speaker Recognition System
        <span class="siamese-badge">Siamese Network</span>
    </h1>
    <p style="font-size: 1.2rem; color: var(--muted); margin-bottom: 2rem;">
        39-Dimension MFCC Features with Deep Learning
    </p>
</div>
""", unsafe_allow_html=True)

# Main tabs
tab1, tab2, tab3 = st.tabs(["🧠 Speaker ID", "📊 Model Info", "📁 File Browser"])

# Tab 1: Speaker Identification
with tab1:
    st.markdown("## 🧠 Speaker Identification")
    st.write("Identify speakers using the Siamese network")
    
    # Load model button
    if not st.session_state.model_loaded:
        if st.button("🚀 Load Siamese Model", type="primary"):
            with st.spinner("Loading Siamese model..."):
                if load_siamese_model():
                    st.success("✅ Model loaded successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to load model")
    
    # Model status
    if st.session_state.model_loaded:
        st.success(f"✅ Siamese model loaded with {len(st.session_state.speaker_embeddings)} speakers")
        
        # Upload audio file
        uploaded_file = st.file_uploader(
            "Upload Audio File",
            type=["wav", "mp3"],
            help="Upload audio file for speaker identification"
        )
        
        # Threshold slider
        threshold = st.slider(
            "Distance Threshold",
            min_value=0.1,
            max_value=2.0,
            value=0.5,
            step=0.1,
            help="Lower threshold = stricter matching"
        )
        
        # Identify button
        if st.button("🧠 Identify Speaker", type="primary", disabled=not uploaded_file):
            if uploaded_file:
                with st.spinner("Analyzing audio..."):
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        # Identify speaker
                        result = identify_speaker(tmp_path, threshold)
                        
                        # Clean up
                        Path(tmp_path).unlink()
                        
                        if "error" in result:
                            st.error(f"❌ {result['error']}")
                        else:
                            st.success(f"✅ Identification Complete!")
                            
                            # Show results
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Identified Speaker", result['speaker'])
                                st.metric("Features Used", result['features_used'])
                            
                            with col2:
                                st.metric("Distance", f"{result['distance']:.4f}")
                                st.metric("Threshold", f"{threshold:.1f}")
                            
                            with col3:
                                st.metric("Match Status", "✅ Match" if result['is_match'] else "❌ No Match")
                                st.metric("Confidence", f"{result['confidence']:.2%}")
                            
                            # Show top matches
                            st.subheader("📊 Top 10 Speaker Matches")
                            sorted_distances = sorted(result['all_distances'].items(), key=lambda x: x[1])
                            
                            for i, (speaker, distance) in enumerate(sorted_distances[:10]):
                                status = "✅ MATCH" if distance < threshold else "❌ NO MATCH"
                                confidence = 1 - distance
                                st.write(f"**{i+1:2d}.** {speaker}: {distance:.4f} ({confidence:6.2%}) {status}")
                    
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

# Tab 2: Model Information
with tab2:
    st.markdown("## 📊 Model Information")
    st.write("Siamese model details and configuration")
    
    if st.session_state.model_loaded:
        st.subheader("🧠 Siamese Network Details")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model Type", "Siamese-like MLP")
            st.metric("Input Shape", "(39,)")
            st.metric("Trained Speakers", len(st.session_state.speaker_embeddings))
        
        with col2:
            st.metric("Model Status", "✅ Loaded")
            st.metric("Framework", "Scikit-learn MLP")
            st.metric("Feature Type", "39-dim MFCC")
        
        # Feature information
        st.subheader("📊 Feature Information")
        st.metric("MFCC Coefficients", "13")
        st.metric("Delta Coefficients", "13")
        st.metric("Delta2 Coefficients", "13")
        st.metric("Total Dimensions", "39")
        
        # Speaker list
        st.subheader("🎯 Trained Speakers")
        speakers = list(st.session_state.speaker_embeddings.keys())
        for i, speaker in enumerate(speakers):
            if i % 5 == 0:
                st.write("")
            st.write(f"✅ {speaker}")
    else:
        st.warning("⚠️ Model not loaded yet")

# Tab 3: File Browser
with tab3:
    st.markdown("## 📁 File Browser")
    st.write("Browse and test audio files")
    
    # Directory input
    directory_path = st.text_input("Directory Path", value="/Users/ashirwadbhardwaj/VRAM_Final /speaker_dataset/50_speakers_audio_data")
    
    if directory_path and Path(directory_path).exists():
        # List speaker directories
        speaker_dirs = [d for d in Path(directory_path).iterdir() if d.is_dir()]
        speaker_dirs.sort()
        
        if speaker_dirs:
            selected_speaker = st.selectbox("Select Speaker", speaker_dirs)
            
            if selected_speaker:
                # List audio files
                audio_files = list(selected_speaker.glob("*.wav")) + list(selected_speaker.glob("*.mp3"))
                audio_files.sort()
                
                if audio_files:
                    selected_file = st.selectbox("Select Audio File", audio_files)
                    
                    if selected_file and st.session_state.model_loaded:
                        if st.button("🧠 Test Selected File", type="primary"):
                            with st.spinner("Analyzing audio..."):
                                result = identify_speaker(str(selected_file))
                                
                                if "error" in result:
                                    st.error(f"❌ {result['error']}")
                                else:
                                    st.success(f"✅ Analysis Complete!")
                                    
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Identified Speaker", result['speaker'])
                                        st.metric("File", selected_file.name)
                                    
                                    with col2:
                                        st.metric("Distance", f"{result['distance']:.4f}")
                                        st.metric("Confidence", f"{result['confidence']:.2%}")
                                    
                                    with col3:
                                        st.metric("Match Status", "✅ Match" if result['is_match'] else "❌ No Match")
                                    
                                    # Show top 5 matches
                                    st.subheader("📊 Top 5 Matches")
                                    sorted_distances = sorted(result['all_distances'].items(), key=lambda x: x[1])
                                    
                                    for i, (speaker, distance) in enumerate(sorted_distances[:5]):
                                        status = "✅ MATCH" if distance < 0.5 else "❌ NO MATCH"
                                        confidence = 1 - distance
                                        st.write(f"**{i+1}.** {speaker}: {distance:.4f} ({confidence:6.2%}) {status}")
                else:
                    st.warning("⚠️ No audio files found")
        else:
            st.warning("⚠️ No speaker directories found")
    else:
        st.warning("⚠️ Directory not found")

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem; color: var(--muted); border-top: 1px solid var(--border); margin-top: 2rem;">
    <p>Speaker Recognition System - Siamese Network</p>
    <p style="font-size: 0.8rem;">39-Dimension MFCC Features with Deep Learning</p>
</div>
""", unsafe_allow_html=True)
