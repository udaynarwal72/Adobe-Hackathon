#!/usr/bin/env python3
"""
Demo script showing how to use the ML PDF Extractor
"""

import os
import sys
import subprocess

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    
    try:
        # Install required packages
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "PyMuPDF", "numpy", "scikit-learn", "nltk", "joblib"], check=True)
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing dependencies: {e}")
        return False

def setup_nltk():
    """Setup NLTK data"""
    print("Setting up NLTK data...")
    
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        print("✓ NLTK data downloaded successfully")
        return True
    except Exception as e:
        print(f"✗ Error setting up NLTK: {e}")
        return False

def run_training():
    """Run the training process"""
    print("\nStarting training process...")
    
    try:
        # Import and run the ML extractor
        sys.path.append('src')
        from ml_pdf_extractor_v2 import MLPDFOutlineExtractor
        
        # Create and train the extractor
        extractor = MLPDFOutlineExtractor()
        extractor.train_models("training_data")
        
        print("✓ Training completed successfully")
        return True
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return False

def test_prediction():
    """Test the trained model"""
    print("\nTesting prediction...")
    
    try:
        sys.path.append('src')
        from ml_pdf_extractor_v2 import MLPDFOutlineExtractor
        
        # Create extractor with trained model
        extractor = MLPDFOutlineExtractor()
        
        # Test on the same PDF
        result = extractor.process_pdf("training_data/pdfs/E0CCG5S312.pdf")
        
        print(f"✓ Prediction completed")
        print(f"Title: {result['title']}")
        print(f"Found {len(result['outline'])} headings:")
        
        for i, heading in enumerate(result['outline'][:5]):
            print(f"  {i+1}. {heading['level']}: {heading['text'][:50]}... (page {heading['page']})")
        
        if len(result['outline']) > 5:
            print(f"  ... and {len(result['outline']) - 5} more headings")
        
        return True
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        return False

def main():
    """Main demo function"""
    print("ML PDF Outline Extractor Demo")
    print("=" * 40)
    
    # Step 1: Install dependencies
    if not install_dependencies():
        return False
    
    # Step 2: Setup NLTK
    if not setup_nltk():
        return False
    
    # Step 3: Check if training data exists
    if not os.path.exists("training_data/pdfs/E0CCG5S312.pdf"):
        print("✗ Training data not found. Please run create_training_example.py first.")
        return False
    
    # Step 4: Train the model
    if not run_training():
        return False
    
    # Step 5: Test prediction
    if not test_prediction():
        return False
    
    print("\n✓ Demo completed successfully!")
    print("\nNext steps:")
    print("1. Add more training examples to improve accuracy")
    print("2. Use the trained model on new PDFs")
    print("3. Fine-tune the model parameters")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
