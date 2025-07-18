#!/bin/bash

# Setup script for ML PDF Extractor

echo "Setting up ML PDF Extractor environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "Installing Python dependencies..."
pip install -r requirements_ml.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p models
mkdir -p training_data/pdfs
mkdir -p training_data/outputs

echo "Setup complete!"
echo ""
echo "To use the ML PDF Extractor:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Place your training PDFs in: training_data/pdfs/"
echo "3. Place corresponding JSON outputs in: training_data/outputs/"
echo "4. Train the model: python src/ml_pdf_extractor.py --mode train --input training_data"
echo "5. Use the model: python src/ml_pdf_extractor.py --mode predict --input input/Pdfs --output output"
