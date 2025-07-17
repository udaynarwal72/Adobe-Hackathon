#!/usr/bin/env python3
"""
Simple test of enhanced PDF processor with app's input PDFs
"""

import os
import sys
import subprocess

def main():
    print("🚀 Processing App Input PDFs with Enhanced Models")
    print("=" * 55)
    
    # Set environment variables to suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # Change to app directory
    app_dir = "/Users/udaynarwal/Projects/Adobe Hackathon/app"
    os.chdir(app_dir)
    
    # Process PDFs from app/input/Pdfs
    input_dir = "input/Pdfs"
    output_dir = "output_enhanced_ml"
    
    print(f"Processing PDFs from: {input_dir}")
    print(f"Output will be saved to: {output_dir}")
    print()
    
    try:
        # Run the enhanced processor
        result = subprocess.run([
            sys.executable, 
            "enhanced_pdf_processor.py",
            "--input", input_dir,
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
            print("\n✅ Processing completed successfully!")
            print(f"\nEnhanced results saved in: {output_dir}/")
            print("Compare with regular output in: output/")
            
            # Show comparison summary
            print("\n📊 COMPARISON SUMMARY:")
            print("-" * 40)
            
            # List files in both directories
            regular_output = "output"
            enhanced_output = output_dir
            
            if os.path.exists(regular_output) and os.path.exists(enhanced_output):
                regular_files = set(f for f in os.listdir(regular_output) if f.endswith('.json'))
                enhanced_files = set(f for f in os.listdir(enhanced_output) if f.endswith('.json'))
                
                common_files = regular_files.intersection(enhanced_files)
                
                for file in sorted(common_files):
                    print(f"📄 {file}")
                    try:
                        # Count headings in regular output
                        regular_path = os.path.join(regular_output, file)
                        enhanced_path = os.path.join(enhanced_output, file)
                        
                        regular_count = subprocess.run(['grep', '-c', '"level":', regular_path], 
                                                     capture_output=True, text=True)
                        enhanced_count = subprocess.run(['grep', '-c', '"level":', enhanced_path], 
                                                      capture_output=True, text=True)
                        
                        reg_num = int(regular_count.stdout.strip()) if regular_count.returncode == 0 else 0
                        enh_num = int(enhanced_count.stdout.strip()) if enhanced_count.returncode == 0 else 0
                        
                        print(f"   Regular model: {reg_num} headings")
                        print(f"   Enhanced model: {enh_num} headings")
                        
                        if enh_num > reg_num:
                            print("   ✅ Enhanced model found more headings")
                        elif enh_num == reg_num:
                            print("   ➡️  Same number of headings")
                        else:
                            print("   ⚠️  Enhanced model found fewer headings")
                        print()
                        
                    except Exception as e:
                        print(f"   Error comparing: {e}")
                        print()
            
        else:
            print(f"\n❌ Processing failed with exit code: {result.returncode}")
            
    except Exception as e:
        print(f"❌ Error running processing: {e}")

if __name__ == "__main__":
    main()
