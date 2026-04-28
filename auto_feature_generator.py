#!/usr/bin/env python3
"""
Automatic Feature Generator
Runs continuously in background to generate features for uploaded speakers
"""

import os
import sys
import json
import time
import threading
import numpy as np
from pathlib import Path
import librosa
import matplotlib.pyplot as plt
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Add current directory to path
sys.path.append('.')

# Import advanced graph generator
from advanced_graph_generator import create_all_advanced_graphs, extract_mfcc_features

class AudioUploadHandler(FileSystemEventHandler):
    """Handle new audio file uploads"""
    
    def __init__(self, output_dir="mfcc_output"):
        self.output_dir = Path(output_dir)
        self.processing_files = set()
        
    def on_created(self, event):
        """Called when a file is created"""
        if event.is_directory:
            return
            
        file_path = Path(event.src_path)
        
        # Check if it's an audio file in uploads directory
        if file_path.parent.name == "uploads" and file_path.suffix.lower() in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']:
            print(f"🎵 New audio file detected: {file_path.name}")
            
            # Start processing in a separate thread
            threading.Thread(target=self.process_audio_file, args=(file_path,), daemon=True).start()
    
    def process_audio_file(self, audio_path):
        """Process uploaded audio file and generate features"""
        try:
            # Avoid processing the same file multiple times
            if str(audio_path) in self.processing_files:
                return
                
            self.processing_files.add(str(audio_path))
            
            print(f"🔄 Processing {audio_path.name}...")
            
            # Extract speaker name from filename
            speaker_name = audio_path.stem
            safe_name = speaker_name.replace(' ', '_').replace('-', '_')
            
            # Check if speaker already has graphs
            speaker_dir = self.output_dir / safe_name
            if speaker_dir.exists():
                existing_graphs = list(speaker_dir.glob("*.png"))
                if len(existing_graphs) >= 9:  # All graphs already exist
                    print(f"✅ {speaker_name} already has {len(existing_graphs)} graphs")
                    self.processing_files.discard(str(audio_path))
                    return
            
            # Extract features
            print(f"🔍 Extracting MFCC features for {speaker_name}...")
            embedding, mfcc, d1, d2, y, sr = extract_mfcc_features(str(audio_path))
            
            if embedding is None:
                print(f"❌ Failed to extract features from {audio_path.name}")
                self.processing_files.discard(str(audio_path))
                return
            
            print(f"✅ Features extracted for {speaker_name}")
            
            # Create speaker directory
            speaker_dir.mkdir(exist_ok=True)
            
            # Generate all graphs
            print(f"📊 Generating comprehensive graphs for {speaker_name}...")
            all_embeddings = []
            graphs = create_all_advanced_graphs(str(audio_path), speaker_name, str(self.output_dir), all_embeddings)
            
            if graphs:
                print(f"✅ Generated {len(graphs)} graphs for {speaker_name}")
                
                # Update embedding JSON
                json_path = speaker_dir / f'{safe_name}_embedding.json'
                existing_data = {}
                
                if json_path.exists():
                    with open(json_path, 'r') as f:
                        existing_data = json.load(f)
                
                # Update with graph information
                existing_data.update({
                    'graphs_created': list(graphs.keys()),
                    'graphs_count': len(graphs),
                    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'auto_generated': True
                })
                
                with open(json_path, 'w') as f:
                    json.dump(existing_data, f, indent=2)
                
                print(f"📁 All features saved for {speaker_name}")
                
                # Trigger UI update notification
                self.notify_ui_update(speaker_name, graphs)
                
            else:
                print(f"❌ Failed to generate graphs for {speaker_name}")
            
        except Exception as e:
            print(f"❌ Error processing {audio_path.name}: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.processing_files.discard(str(audio_path))
    
    def notify_ui_update(self, speaker_name, graphs):
        """Notify UI about new features (could be implemented with WebSocket or file flag)"""
        # Create a notification file that UI can check
        notification_file = self.output_dir / "ui_update_notification.json"
        
        try:
            notification_data = {
                'speaker': speaker_name,
                'graphs_count': len(graphs),
                'graphs_created': list(graphs.keys()),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'status': 'completed'
            }
            
            with open(notification_file, 'w') as f:
                json.dump(notification_data, f, indent=2)
                
            print(f"📢 UI notification created for {speaker_name}")
            
        except Exception as e:
            print(f"❌ Error creating UI notification: {e}")

class AutoFeatureGenerator:
    """Automatic feature generator that runs continuously"""
    
    def __init__(self, uploads_dir="uploads", output_dir="mfcc_output"):
        self.uploads_dir = Path(uploads_dir)
        self.output_dir = Path(output_dir)
        self.observer = None
        self.running = False
        
    def start(self):
        """Start the automatic feature generator"""
        print("🚀 Starting Automatic Feature Generator...")
        
        # Create directories if they don't exist
        self.uploads_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up file system observer
        event_handler = AudioUploadHandler(self.output_dir)
        self.observer = Observer()
        self.observer.schedule(event_handler, str(self.uploads_dir), recursive=True)
        
        # Start observer
        self.observer.start()
        self.running = True
        
        print(f"👀 Monitoring uploads directory: {self.uploads_dir}")
        print(f"📂 Output directory: {self.output_dir}")
        print("✅ Auto Feature Generator is running...")
        
        # Process existing files that don't have graphs yet
        self.process_existing_files()
        
        # Keep the script running
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⏹️  Stopping Auto Feature Generator...")
            self.stop()
    
    def process_existing_files(self):
        """Process existing audio files that don't have graphs yet"""
        print("🔍 Checking for existing files that need processing...")
        
        audio_files = []
        for ext in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']:
            audio_files.extend(self.uploads_dir.glob(f"*{ext}"))
        
        if not audio_files:
            print("📁 No existing audio files found")
            return
        
        print(f"📁 Found {len(audio_files)} existing audio files")
        
        for audio_file in audio_files:
            speaker_name = audio_file.stem
            safe_name = speaker_name.replace(' ', '_').replace('-', '_')
            speaker_dir = self.output_dir / safe_name
            
            # Check if speaker already has complete graphs
            if speaker_dir.exists():
                existing_graphs = list(speaker_dir.glob("*.png"))
                if len(existing_graphs) >= 9:
                    print(f"✅ {speaker_name} already has {len(existing_graphs)} graphs")
                    continue
            
            # Process this file
            print(f"🔄 Processing existing file: {audio_file.name}")
            event_handler = AudioUploadHandler(self.output_dir)
            event_handler.process_audio_file(audio_file)
    
    def stop(self):
        """Stop the automatic feature generator"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
        self.running = False
        print("✅ Auto Feature Generator stopped")

def main():
    """Main function to run the auto feature generator"""
    print("=" * 60)
    print("🎵 AUTOMATIC FEATURE GENERATOR")
    print("Continuously monitors uploads and generates voice features")
    print("=" * 60)
    
    generator = AutoFeatureGenerator()
    
    try:
        generator.start()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        generator.stop()

if __name__ == "__main__":
    main()
