#!/usr/bin/env python3
"""
Test the enhanced PDF processor with Challenge-1(a) trained models
"""

import os
import sys
import subprocess

def main():
    print("🧪 Testing Enhanced PDF Processor with Challenge-1(a) Models")
    print("=" * 60)
    
    # Set environment variables to suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # Change to app directory
    app_dir = "/Users/mk941/Downloads/Desktop/Adobe Hackathon/Adobe-Hackathon/app"
    os.chdir(app_dir)
    
    # Test with Challenge-1(a) PDFs to validate the training
    challenge_pdfs = "/Users/mk941/Downloads/Desktop/Adobe Hackathon/Adobe-Hackathon/Challenge - 1(a)/Datasets/Pdfs"
    output_dir = "output_challenge_test"
    
    print(f"Testing with PDFs from: {challenge_pdfs}")
    print(f"Output will be saved to: {output_dir}")
    print()
    
    try:
        # Run the enhanced processor
        result = subprocess.run([
            sys.executable, 
            "enhanced_pdf_processor.py",
            "--input", challenge_pdfs,
            "--output", output_dir,
            "--ensemble"
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
            print("\n✅ Testing completed successfully!")
            print(f"\nResults saved in: {output_dir}/")
            print("You can compare these results with the original ground truth")
        else:
            print(f"\n❌ Testing failed with exit code: {result.returncode}")
            
    except Exception as e:
        print(f"❌ Error running test: {e}")

if __name__ == "__main__":
    main()
