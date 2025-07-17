#!/usr/bin/env python3
"""
Simple wrapper to train models with Challenge-1(a) data
"""

import os
import sys
import subprocess

def main():
    print("🚀 Training ML models with Challenge-1(a) ground truth data...")
    print("=" * 60)
    
    # Set environment variables to suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # Change to app directory
    app_dir = "/Users/udaynarwal/Projects/Adobe Hackathon/app"
    os.chdir(app_dir)
    
    try:
        # Run the training script
        result = subprocess.run([
            sys.executable, 
            "src/train_with_challenge_data.py"
        ], capture_output=True, text=True)
        
        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            # Filter out TensorFlow warnings
            stderr_lines = result.stderr.split('\n')
            filtered_lines = []
            for line in stderr_lines:
                if not any(warning in line.lower() for warning in 
                          ['warning', 'deprecated', 'futurewarning', 'tensorflow']):
                    filtered_lines.append(line)
            filtered_stderr = '\n'.join(filtered_lines).strip()
            if filtered_stderr:
                print(filtered_stderr)
        
        if result.returncode == 0:
            print("\n✅ Training completed successfully!")
            print("\nModels saved in: app/models/")
            print("- rf_model_challenge.pkl (Random Forest)")
            print("- deep_model_challenge.h5 (Deep Learning)")
            print("- training_summary_challenge.json (Summary)")
        else:
            print(f"\n❌ Training failed with exit code: {result.returncode}")
            
    except Exception as e:
        print(f"❌ Error running training: {e}")

if __name__ == "__main__":
    main()
