#!/usr/bin/env python3
"""
Data preparation script for ML PDF Extractor
This script helps organize your training data and validates the format
"""

import os
import json
import shutil
from pathlib import Path
import argparse

def validate_json_format(json_path):
    """Validate that JSON file has the correct format"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check required keys
        if 'title' not in data:
            return False, "Missing 'title' key"
        
        if 'outline' not in data:
            return False, "Missing 'outline' key"
        
        # Check outline format
        outline = data['outline']
        if not isinstance(outline, list):
            return False, "'outline' must be a list"
        
        for i, item in enumerate(outline):
            if not isinstance(item, dict):
                return False, f"Outline item {i} must be a dictionary"
            
            required_keys = ['level', 'text', 'page']
            for key in required_keys:
                if key not in item:
                    return False, f"Outline item {i} missing '{key}' key"
            
            # Validate level format
            if not item['level'].startswith('H') or not item['level'][1:].isdigit():
                return False, f"Invalid level format: {item['level']} (should be H1, H2, etc.)"
            
            # Validate page is a number
            if not isinstance(item['page'], int):
                return False, f"Page must be an integer, got {type(item['page'])}"
        
        return True, "Valid format"
        
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except Exception as e:
        return False, f"Error reading file: {e}"

def prepare_training_data(source_dir, target_dir):
    """Prepare training data by organizing PDFs and JSON files"""
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Create target directories
    pdf_dir = target_path / "pdfs"
    json_dir = target_path / "outputs"
    
    pdf_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all PDF files
    pdf_files = list(source_path.glob("**/*.pdf"))
    json_files = list(source_path.glob("**/*.json"))
    
    print(f"Found {len(pdf_files)} PDF files and {len(json_files)} JSON files")
    
    # Create mapping of base names
    pdf_mapping = {pdf.stem: pdf for pdf in pdf_files}
    json_mapping = {json.stem: json for json in json_files}
    
    # Process matching pairs
    processed_count = 0
    errors = []
    
    for base_name in pdf_mapping:
        if base_name in json_mapping:
            pdf_file = pdf_mapping[base_name]
            json_file = json_mapping[base_name]
            
            # Validate JSON format
            is_valid, message = validate_json_format(json_file)
            if not is_valid:
                errors.append(f"{json_file.name}: {message}")
                continue
            
            # Copy files to target directory
            target_pdf = pdf_dir / f"{base_name}.pdf"
            target_json = json_dir / f"{base_name}.json"
            
            shutil.copy2(pdf_file, target_pdf)
            shutil.copy2(json_file, target_json)
            
            processed_count += 1
            print(f"✓ Processed: {base_name}")
        else:
            print(f"⚠ No matching JSON for PDF: {base_name}")
    
    print(f"\nProcessed {processed_count} file pairs")
    
    if errors:
        print(f"\nErrors found in {len(errors)} files:")
        for error in errors:
            print(f"  - {error}")
    
    return processed_count, errors

def create_sample_json():
    """Create a sample JSON file showing the expected format"""
    sample = {
        "title": "Sample Document Title",
        "outline": [
            {
                "level": "H1",
                "text": "Introduction",
                "page": 1
            },
            {
                "level": "H2",
                "text": "Overview",
                "page": 2
            },
            {
                "level": "H3",
                "text": "Key Points",
                "page": 3
            }
        ]
    }
    
    with open("sample_output_format.json", "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    
    print("Created sample_output_format.json showing expected format")

def main():
    parser = argparse.ArgumentParser(description='Prepare training data for ML PDF Extractor')
    parser.add_argument('--source', required=True, help='Source directory containing PDFs and JSON files')
    parser.add_argument('--target', required=True, help='Target directory to organize training data')
    parser.add_argument('--validate-only', action='store_true', help='Only validate JSON files without copying')
    parser.add_argument('--create-sample', action='store_true', help='Create sample JSON format file')
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_json()
        return
    
    if not os.path.exists(args.source):
        print(f"Error: Source directory '{args.source}' does not exist")
        return
    
    if args.validate_only:
        json_files = list(Path(args.source).glob("**/*.json"))
        errors = []
        
        for json_file in json_files:
            is_valid, message = validate_json_format(json_file)
            if not is_valid:
                errors.append(f"{json_file.name}: {message}")
            else:
                print(f"✓ Valid: {json_file.name}")
        
        if errors:
            print(f"\nErrors found in {len(errors)} files:")
            for error in errors:
                print(f"  - {error}")
        else:
            print("\nAll JSON files are valid!")
    else:
        processed_count, errors = prepare_training_data(args.source, args.target)
        
        if processed_count == 0:
            print("No matching PDF-JSON pairs found!")
        else:
            print(f"\nTraining data prepared successfully!")
            print(f"Place your training files in:")
            print(f"  PDFs: {args.target}/pdfs/")
            print(f"  JSON: {args.target}/outputs/")

if __name__ == "__main__":
    main()
