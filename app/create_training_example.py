#!/usr/bin/env python3
"""
Quick test script to create training data from your current PDF and expected output
"""

import json
import os
import sys
from pathlib import Path

def create_training_example():
    """Create a training example from your current PDF and expected output"""
    
    # Your expected output
    expected_output = {
        "title": "Overview  Foundation Level Extensions  ",
        "outline": [
            {
                "level": "H1",
                "text": "Revision History ",
                "page": 2
            },
            {
                "level": "H1",
                "text": "Table of Contents ",
                "page": 3
            },
            {
                "level": "H1",
                "text": "Acknowledgements ",
                "page": 4
            },
            {
                "level": "H1",
                "text": "1. Introduction to the Foundation Level Extensions ",
                "page": 5
            },
            {
                "level": "H1",
                "text": "2. Introduction to Foundation Level Agile Tester Extension ",
                "page": 6
            },
            {
                "level": "H2",
                "text": "2.1 Intended Audience ",
                "page": 6
            },
            {
                "level": "H2",
                "text": "2.2 Career Paths for Testers ",
                "page": 6
            },
            {
                "level": "H2",
                "text": "2.3 Learning Objectives ",
                "page": 6
            },
            {
                "level": "H2",
                "text": "2.4 Entry Requirements ",
                "page": 7
            },
            {
                "level": "H2",
                "text": "2.5 Structure and Course Duration ",
                "page": 7
            },
            {
                "level": "H2",
                "text": "2.6 Keeping It Current ",
                "page": 8
            },
            {
                "level": "H1",
                "text": "3. Overview of the Foundation Level Extension – Agile TesterSyllabus ",
                "page": 9
            },
            {
                "level": "H2",
                "text": "3.1 Business Outcomes ",
                "page": 9
            },
            {
                "level": "H2",
                "text": "3.2 Content ",
                "page": 9
            },
            {
                "level": "H1",
                "text": "4. References ",
                "page": 11
            },
            {
                "level": "H2",
                "text": "4.1 Trademarks ",
                "page": 11
            },
            {
                "level": "H2",
                "text": "4.2 Documents and Web Sites ",
                "page": 11
            }
        ]
    }
    
    # Create training data directories
    training_dir = Path("training_data")
    pdf_dir = training_dir / "pdfs"
    json_dir = training_dir / "outputs"
    
    pdf_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy the PDF file
    source_pdf = Path("input/Pdfs/E0CCG5S312.pdf")
    target_pdf = pdf_dir / "E0CCG5S312.pdf"
    
    if source_pdf.exists():
        import shutil
        shutil.copy2(source_pdf, target_pdf)
        print(f"✓ Copied PDF: {target_pdf}")
    else:
        print(f"✗ Source PDF not found: {source_pdf}")
        return False
    
    # Create the JSON file
    target_json = json_dir / "E0CCG5S312.json"
    with open(target_json, 'w', encoding='utf-8') as f:
        json.dump(expected_output, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Created JSON: {target_json}")
    
    print("\nTraining data created successfully!")
    print("You can now train the model with:")
    print("python src/ml_pdf_extractor_v2.py --mode train --input training_data --verbose")
    
    return True

if __name__ == "__main__":
    create_training_example()
