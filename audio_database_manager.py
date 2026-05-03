#!/usr/bin/env python3
"""
Audio Database Manager
A standalone script for uploading, storing, and comparing audio files
"""

import os
import json
import shutil
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from extract_mfcc import MFCCExtractor
from advanced_graph_generator import create_all_advanced_graphs

class AudioDatabaseManager:
    """Manages audio file uploads, storage, and comparison"""
    
    def __init__(self, db_path="audio_database"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.audio_dir = self.db_path / "audio_files"
        self.audio_dir.mkdir(exist_ok=True)
        
        # Create output directories for graphs
        self.all_output_dir = Path("all_output")
        self.all_output_dir.mkdir(exist_ok=True)
        self.option_1_folder = self.all_output_dir / "option_1_folder"
        self.option_1_folder.mkdir(exist_ok=True)
        self.option_2_folder = self.all_output_dir / "option_2_folder"
        self.option_2_folder.mkdir(exist_ok=True)
        
        # Database files
        self.embeddings_file = self.db_path / "embeddings.json"
        self.metadata_file = self.db_path / "metadata.json"
        
        # Initialize feature extractor
        self.extractor = MFCCExtractor()
        
        # Load existing database
        self.embeddings = self.load_embeddings()
        self.metadata = self.load_metadata()
        
        # Load Siamese model
        self.encoder = None
        self.scaler = None
        self.siamese_embeddings = {}
        self.load_siamese_model()
        
        print(f"📁 Database initialized at: {self.db_path}")
        print(f"🎵 Audio files stored at: {self.audio_dir}")
        print(f"📊 Current entries: {len(self.embeddings)}")
        print(f"📈 Graph output: {self.all_output_dir}")
    
    def load_siamese_model(self):
        """Load the Siamese model for enhanced comparison"""
        try:
            model_dir = Path("siamese_model_39dim")
            
            if not model_dir.exists():
                print("⚠️  Siamese model not found - using simple comparison only")
                return False
            
            # Load encoder
            with open(model_dir / "encoder.pkl", 'rb') as f:
                self.encoder = pickle.load(f)
            
            # Load scaler
            with open(model_dir / "scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load speaker embeddings
            with open(model_dir / "speaker_embeddings_39dim.json", 'r') as f:
                embeddings_data = json.load(f)
            
            for speaker_name, embedding_list in embeddings_data.items():
                self.siamese_embeddings[speaker_name] = np.array(embedding_list)
            
            print(f"🧠 Siamese model loaded: {len(self.siamese_embeddings)} trained speakers")
            return True
            
        except Exception as e:
            print(f"⚠️  Could not load Siamese model: {e}")
            return False
    
    def load_embeddings(self):
        """Load existing embeddings from file"""
        if self.embeddings_file.exists():
            with open(self.embeddings_file, 'r') as f:
                data = json.load(f)
                # Convert lists back to numpy arrays
                return {name: np.array(embedding) for name, embedding in data.items()}
        return {}
    
    def load_metadata(self):
        """Load existing metadata from file"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_embeddings(self):
        """Save embeddings to file"""
        # Convert numpy arrays to lists for JSON serialization
        data = {name: embedding.tolist() for name, embedding in self.embeddings.items()}
        with open(self.embeddings_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save_metadata(self):
        """Save metadata to file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def upload_audio(self):
        """Upload and process a new audio file"""
        print("\n" + "="*50)
        print("🎵 AUDIO UPLOAD")
        print("="*50)
        
        # Get audio path
        audio_path = input("Enter path to audio file: ").strip()
        
        if not os.path.exists(audio_path):
            print("❌ File not found!")
            return False
        
        # Get speaker name
        speaker_name = input("Enter speaker name: ").strip()
        if not speaker_name:
            print("❌ Speaker name cannot be empty!")
            return False
        
        try:
            # Extract features
            print(f"🔄 Processing audio: {audio_path}")
            embedding, mfcc, delta, delta2, y, sr = self.extractor.extract_ui_features(audio_path)
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_ext = Path(audio_path).suffix
            audio_filename = f"{speaker_name}_{timestamp}{file_ext}"
            audio_save_path = self.audio_dir / audio_filename
            
            # Copy audio file to database
            shutil.copy2(audio_path, audio_save_path)
            
            # Store embedding and metadata
            self.embeddings[speaker_name] = embedding
            self.metadata[speaker_name] = {
                "audio_file": audio_filename,
                "upload_time": datetime.now().isoformat(),
                "original_path": audio_path,
                "feature_shape": list(embedding.shape),
                "sample_rate": sr,
                "duration": len(y) / sr
            }
            
            # Save database
            self.save_embeddings()
            self.save_metadata()
            
            print(f"✅ Audio uploaded successfully!")
            print(f"📁 Saved as: {audio_filename}")
            print(f"🎯 Speaker: {speaker_name}")
            print(f"📊 Features extracted: {embedding.shape}")
            print(f"⏱️ Duration: {len(y) / sr:.2f} seconds")
            
            # Generate advanced graphs for option 1
            print(f"\n📈 Generating advanced graphs for uploaded audio...")
            try:
                # Convert embeddings dict to list for graph generator (excluding current speaker)
                all_embeddings_list = [v for k, v in self.embeddings.items() if k != speaker_name]
                graphs = create_all_advanced_graphs(
                    audio_path=str(audio_save_path),
                    speaker_name=speaker_name,
                    output_dir=str(self.option_1_folder),
                    all_embeddings=all_embeddings_list
                )
                if graphs:
                    print(f"✅ Advanced graphs generated successfully in: {self.option_1_folder}/{speaker_name}")
                    for graph_type, path in graphs.items():
                        print(f"   📊 {graph_type}: {path}")
                else:
                    print(f"⚠️ Could not generate graphs")
            except Exception as e:
                print(f"⚠️ Error generating graphs: {e}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error processing audio: {e}")
            return False
    
    def compare_audio(self):
        """Compare an audio file against the database"""
        print("\n" + "="*50)
        print("🔍 AUDIO COMPARISON")
        print("="*50)
        
        if not self.embeddings:
            print("❌ No audio files in database! Upload some audio first.")
            return
        
        # Show available speakers
        print("📋 Available speakers in database:")
        for i, speaker in enumerate(self.embeddings.keys(), 1):
            duration = self.metadata[speaker]["duration"]
            print(f"  {i}. {speaker} ({duration:.2f}s)")
        
        # Get audio path
        audio_path = input("\nEnter path to audio file to compare: ").strip()
        
        if not os.path.exists(audio_path):
            print("❌ File not found!")
            return
        
        try:
            # Extract features from comparison audio
            print(f"🔄 Processing comparison audio: {audio_path}")
            embedding, mfcc, delta, delta2, y, sr = self.extractor.extract_ui_features(audio_path)
            
            # Calculate MULTIPLE metrics for better discrimination
            # Including: Cosine, Euclidean, Manhattan, AND Siamese model
            results = {}
            for speaker_name, stored_embedding in self.embeddings.items():
                # 1. Cosine similarity (direction match)
                cosine_sim = np.dot(embedding, stored_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(stored_embedding)
                )
                
                # 2. Euclidean distance (magnitude match)
                euclidean_dist = np.linalg.norm(embedding - stored_embedding)
                
                # 3. Manhattan distance (L1 norm)
                manhattan_dist = np.sum(np.abs(embedding - stored_embedding))
                
                # 4. SIANESE MODEL comparison (if available)
                siamese_sim = 0.0
                if self.encoder is not None and self.scaler is not None:
                    try:
                        # Calculate feature difference
                        diff = np.abs(embedding - stored_embedding)
                        # Scale the difference
                        diff_scaled = self.scaler.transform([diff])
                        # Predict similarity using trained encoder
                        similarity_prob = self.encoder.predict_proba(diff_scaled)[0]
                        siamese_sim = similarity_prob[1]  # Probability of being same speaker
                    except Exception as e:
                        siamese_sim = 0.0
                
                # 5. Combined score (weighted)
                # Lower distance = better match, higher cosine/siamese = better match
                # Normalize distances (assuming typical range 0-50)
                norm_euclidean = max(0, 1 - (euclidean_dist / 50))
                norm_manhattan = max(0, 1 - (manhattan_dist / 500))
                
                # Combined score with Siamese model (weighted ensemble)
                combined_score = (
                    0.35 * cosine_sim + 
                    0.25 * norm_euclidean + 
                    0.10 * norm_manhattan +
                    0.30 * siamese_sim  # Siamese model has significant weight
                )
                
                results[speaker_name] = {
                    'cosine': cosine_sim,
                    'euclidean': euclidean_dist,
                    'manhattan': manhattan_dist,
                    'siamese': siamese_sim,
                    'combined': combined_score
                }
            
            # Sort by combined score (highest first)
            sorted_speakers = sorted(results.items(), key=lambda x: x[1]['combined'], reverse=True)
            
            print(f"\n🎯 COMPARISON RESULTS:")
            print("="*90)
            print(f"{'Rank':<6} {'Speaker':<18} {'Cosine':<9} {'Euclidean':<11} {'Siamese':<9} {'Combined':<10} {'Status'}")
            print("="*90)
            
            # Calculate average similarity for unknown detection
            all_cosines = [r['cosine'] for r in results.values()]
            avg_cosine = np.mean(all_cosines)
            std_cosine = np.std(all_cosines)
            
            for i, (speaker, metrics) in enumerate(sorted_speakers, 1):
                # STRICT matching criteria to prevent false positives
                is_match = (metrics['cosine'] > 0.998 and 
                           metrics['euclidean'] < 0.5 and 
                           metrics['siamese'] > 0.99 and  # Very high Siamese threshold
                           metrics['combined'] > 0.95)
                
                match_status = "✅ MATCH" if is_match else "❌ NO MATCH"
                siamese_str = f"{metrics['siamese']:.4f}" if metrics['siamese'] > 0 else "N/A"
                print(f"{i:<6} {speaker:<18} {metrics['cosine']:<9.4f} {metrics['euclidean']:<11.2f} {siamese_str:<9} {metrics['combined']:<10.4f} {match_status}")
            
            print("="*90)
            
            # UNKNOWN SPEAKER DETECTION
            best_match = sorted_speakers[0]
            second_best = sorted_speakers[1] if len(sorted_speakers) > 1 else (None, {'combined': 0})
            
            # Calculate confidence metrics
            gap = best_match[1]['combined'] - second_best[1]['combined']
            is_clear_winner = gap > 0.15  # Larger gap for confidence
            is_high_similarity = best_match[1]['cosine'] > 0.998
            is_low_distance = best_match[1]['euclidean'] < 0.5
            is_high_siamese = best_match[1]['siamese'] > 0.99 if best_match[1]['siamese'] > 0 else False
            
            print(f"\n📊 ANALYSIS:")
            print(f"   Best match: {best_match[0]} (Combined: {best_match[1]['combined']:.4f})")
            print(f"   Second best: {second_best[0]} (Combined: {second_best[1]['combined']:.4f})")
            print(f"   Gap: {gap:.4f} (Need >0.15 for clear winner)")
            print(f"   Cosine similarity: {best_match[1]['cosine']:.4f} (Need > 0.998)")
            print(f"   Euclidean distance: {best_match[1]['euclidean']:.4f} (Need < 0.5)")
            if best_match[1]['siamese'] > 0:
                print(f"   Siamese model: {best_match[1]['siamese']:.4f} (Need > 0.99)")
                print(f"   🧠 Using trained neural network for verification")
            
            # Debug: Show which conditions pass/fail
            print(f"\n🔍 CONDITIONS CHECK:")
            print(f"   Cosine > 0.998? {best_match[1]['cosine']:.4f} > 0.998 = {is_high_similarity}")
            print(f"   Euclidean < 0.5? {best_match[1]['euclidean']:.4f} < 0.5 = {is_low_distance}")
            print(f"   Siamese > 0.99? {best_match[1]['siamese']:.4f} > 0.99 = {is_high_siamese}")
            print(f"   Gap > 0.15? {gap:.4f} > 0.15 = {is_clear_winner}")
            
            # Decision logic with strict Siamese model thresholds
            # Special case: Perfect Siamese match (1.0) means it's definitely the speaker
            is_perfect_siamese = best_match[1]['siamese'] >= 0.999
            
            if is_high_similarity and is_low_distance and is_high_siamese and (is_clear_winner or is_perfect_siamese):
                print(f"\n🏆 CONFIDENT MATCH: {best_match[0]}")
                print(f"   ✅ Euclidean: {best_match[1]['euclidean']:.4f} (< 0.5)")
                print(f"   ✅ Cosine: {best_match[1]['cosine']:.4f} (> 0.998)") 
                print(f"   ✅ Siamese: {best_match[1]['siamese']:.4f} (> 0.99)")
                print(f"   ✅ Combined: {best_match[1]['combined']:.4f} (> 0.95)")
                if is_perfect_siamese and not is_clear_winner:
                    print(f"   ℹ️  Note: Gap was small but Siamese is perfect (≥0.999)")
                print(f"   This speaker is VERIFIED in the database!")
            elif best_match[1]['siamese'] > 0.5 and best_match[1]['siamese'] < 0.99:
                print(f"\n❓ BORDERLINE MATCH: {best_match[0]}")
                print(f"   Siamese model shows some similarity ({best_match[1]['siamese']:.4f})")
                print(f"   but not confident enough (> 0.99 required).")
                print(f"   This is likely an UNKNOWN SPEAKER or needs verification.")
            elif best_match[1]['siamese'] > 0 and best_match[1]['siamese'] <= 0.5:
                print(f"\n🚫 UNKNOWN SPEAKER")
                print(f"   Siamese model confidence too low: {best_match[1]['siamese']:.4f}")
                print(f"   Required: > 0.99 for verification")
                print(f"   This speaker is NOT in the database!")
            else:
                print(f"\n🚫 UNKNOWN SPEAKER")
                print(f"   No matching speaker found in database!")
                print(f"   Best attempt: {best_match[0]} with {best_match[1]['cosine']:.4f} similarity")
            
            # Ask for name to save comparison graphs
            save_graphs = input("\nDo you want to generate graphs for this comparison? (yes/no): ").strip().lower()
            if save_graphs in ['yes', 'y']:
                compare_name = input("Enter a name for this comparison (e.g., 'Test_Speaker_35'): ").strip()
                if not compare_name:
                    compare_name = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                print(f"\n📈 Generating advanced graphs for comparison...")
                try:
                    # Convert embeddings dict to list for graph generator
                    all_embeddings_list = list(self.embeddings.values())
                    graphs = create_all_advanced_graphs(
                        audio_path=audio_path,
                        speaker_name=compare_name,
                        output_dir=str(self.option_2_folder),
                        all_embeddings=all_embeddings_list
                    )
                    if graphs:
                        print(f"✅ Comparison graphs generated successfully in: {self.option_2_folder}/{compare_name}")
                        for graph_type, path in graphs.items():
                            print(f"   📊 {graph_type}: {path}")
                    else:
                        print(f"⚠️ Could not generate graphs")
                except Exception as e:
                    print(f"⚠️ Error generating graphs: {e}")
            
        except Exception as e:
            print(f"❌ Error comparing audio: {e}")
    
    def list_database(self):
        """List all entries in the database"""
        print("\n" + "="*50)
        print("📋 DATABASE CONTENTS")
        print("="*50)
        
        if not self.embeddings:
            print("📭 Database is empty!")
            return
        
        print(f"📊 Total entries: {len(self.embeddings)}")
        print()
        
        for speaker_name in self.embeddings.keys():
            metadata = self.metadata[speaker_name]
            print(f"🎤 Speaker: {speaker_name}")
            print(f"   📁 File: {metadata['audio_file']}")
            print(f"   ⏱️ Duration: {metadata['duration']:.2f}s")
            print(f"   📅 Uploaded: {metadata['upload_time'][:19]}")
            print(f"   📊 Features: {metadata['feature_shape']}")
            print()
    
    def run(self):
        """Main interactive loop"""
        print("🎵 AUDIO DATABASE MANAGER")
        print("="*50)
        print("Manage your audio database for speaker recognition")
        
        while True:
            print("\n" + "="*30)
            print("📋 MAIN MENU")
            print("="*30)
            print("1. 🎵 Upload new audio")
            print("2. 🔍 Compare audio")
            print("3. 📋 List database")
            print("4. ❌ Exit")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                self.upload_audio()
            elif choice == "2":
                self.compare_audio()
            elif choice == "3":
                self.list_database()
            elif choice == "4":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice! Please enter 1-4.")

def main():
    """Main function"""
    try:
        manager = AudioDatabaseManager()
        manager.run()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
