#!/usr/bin/env python3
"""
Simple wrapper script for ML PDF Extractor
Suppresses TensorFlow warnings and provides easy-to-use interface
"""

import os
import sys
import warnings
import subprocess

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

def main():
    """Simple wrapper to run ML PDF extractor with cleaner output"""
    
    if len(sys.argv) < 2:
        print("🤖 ML PDF Outline Extractor")
        print("=" * 40)
        print()
        print("Usage:")
        print("  python run_ml_extractor.py train    # Train the model")
        print("  python run_ml_extractor.py predict  # Run predictions")
        print("  python run_ml_extractor.py help     # Show detailed help")
        print()
        print("Examples:")
        print("  python run_ml_extractor.py train")
        print("  python run_ml_extractor.py predict")
        return

    command = sys.argv[1].lower()
    
    if command == "help":
        # Show detailed help
        subprocess.run([sys.executable, "src/ml_pdf_extractor.py", "--help"])
        return
    
    elif command == "train":
        print("🤖 Training ML models...")
        print("This will train both Random Forest and Deep Learning models")
        print()
        
        # Run training
        result = subprocess.run([
            sys.executable, "src/ml_pdf_extractor.py",
            "--mode", "train",
            "--input", "training_data",
            "--model-path", "models",
            "--verbose"
        ], capture_output=False)
        
        if result.returncode == 0:
            print("\n✅ Training completed successfully!")
            print("Models saved in 'models/' directory")
        else:
            print("\n❌ Training failed!")
            
    elif command == "predict":
        print("🤖 Running ML predictions...")
        print("Processing PDFs in 'input/Pdfs/' directory")
        print()
        
        # Create output directory
        os.makedirs("output_ml", exist_ok=True)
        
        # Run predictions
        result = subprocess.run([
            sys.executable, "src/ml_pdf_extractor.py",
            "--mode", "predict",
            "--input", "input/Pdfs",
            "--output", "output_ml",
            "--model-path", "models",
            "--verbose"
        ], capture_output=False)
        
        if result.returncode == 0:
            print("\n✅ Predictions completed successfully!")
            print("Results saved in 'output_ml/' directory")
        else:
            print("\n❌ Prediction failed!")
    
    else:
        print(f"❌ Unknown command: {command}")
        print("Use 'train', 'predict', or 'help'")

if __name__ == "__main__":
    main()
