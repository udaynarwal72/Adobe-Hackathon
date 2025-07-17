#!/usr/bin/env python3
"""
Simple Enhanced PDF Processor
Clean interface for processing PDFs with Challenge-1(a) trained models
"""

import os
import sys
import subprocess
import argparse

def suppress_warnings():
    """Suppress TensorFlow warnings for cleaner output"""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def process_pdf(input_path, output_path=None, ensemble=True):
    """Process a PDF with enhanced models"""
    suppress_warnings()
    
    # Build command
    cmd = [
        sys.executable, 
        "enhanced_pdf_processor.py",
        "--input", input_path
    ]
    
    if output_path:
        cmd.extend(["--output", output_path])
    
    if ensemble:
        cmd.append("--ensemble")
    else:
        cmd.append("--deep-only")
    
    # Run command
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Print only relevant output (filter TensorFlow warnings)
        if result.stdout:
            lines = result.stdout.split('\n')
            filtered_lines = []
            for line in lines:
                if 'Models loaded successfully' in line or \
                   'Processing:' in line or \
                   'Extracted' in line or \
                   'Identified' in line or \
                   'Results saved' in line or \
                   'Found' in line or \
                   'PROCESSING COMPLETE' in line or \
                   '✅' in line or '❌' in line:
                    filtered_lines.append(line)
            
            if filtered_lines:
                print('\n'.join(filtered_lines))
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced PDF Processor with Challenge-1(a) trained models",
        epilog="""
Examples:
  %(prog)s my_document.pdf                    # Process single PDF
  %(prog)s my_document.pdf -o result.json     # Specify output file  
  %(prog)s pdf_folder/                        # Process all PDFs in folder
  %(prog)s my_document.pdf --deep-only        # Use only deep learning model
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("input", 
                       help="PDF file or directory to process")
    parser.add_argument("-o", "--output", 
                       help="Output JSON file or directory")
    parser.add_argument("--deep-only", action="store_true",
                       help="Use deep learning model only (default: ensemble)")
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input):
        print(f"❌ Error: {args.input} does not exist")
        return 1
    
    print("🚀 Enhanced PDF Processor with Challenge-1(a) Models")
    print("=" * 55)
    
    # Process
    success = process_pdf(args.input, args.output, ensemble=not args.deep_only)
    
    if success:
        print("\n✅ Processing completed successfully!")
        if args.output:
            print(f"📄 Results saved to: {args.output}")
        return 0
    else:
        print("\n❌ Processing failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
