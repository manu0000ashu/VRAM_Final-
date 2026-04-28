#!/usr/bin/env python3
"""
Start the Automatic Feature Generator
This script starts the background process that monitors uploads and generates features
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def start_auto_generator():
    """Start the auto feature generator in background"""
    print("🚀 Starting Automatic Feature Generator...")
    
    # Check if auto_feature_generator.py exists
    script_path = Path("auto_feature_generator.py")
    if not script_path.exists():
        print("❌ auto_feature_generator.py not found!")
        return False
    
    try:
        # Start the auto generator in background
        process = subprocess.Popen([sys.executable, str(script_path)], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE,
                                 text=True)
        
        print(f"✅ Auto Feature Generator started (PID: {process.pid})")
        print("📁 Monitoring uploads/ directory for new audio files...")
        print("🔄 Features will be generated automatically in the background")
        print("⏹️  Press Ctrl+C to stop the generator")
        
        # Wait a moment to see if it starts properly
        time.sleep(2)
        
        # Check if process is still running
        if process.poll() is None:
            print("✅ Auto Feature Generator is running successfully!")
            return True
        else:
            # Process died, show error
            stdout, stderr = process.communicate()
            print("❌ Auto Feature Generator failed to start!")
            if stderr:
                print(f"Error: {stderr}")
            if stdout:
                print(f"Output: {stdout}")
            return False
            
    except Exception as e:
        print(f"❌ Error starting Auto Feature Generator: {e}")
        return False

if __name__ == "__main__":
    success = start_auto_generator()
    if success:
        print("\n🎯 The Auto Feature Generator is now running!")
        print("📱 Upload audio files through the Streamlit UI and features will be generated automatically")
        print("🔍 Check the 'Voice Features' tab to see generated graphs")
        
        # Keep this script running
        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            print("\n⏹️  Stopping Auto Feature Generator...")
    else:
        print("❌ Failed to start Auto Feature Generator")
        sys.exit(1)
