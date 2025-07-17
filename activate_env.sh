#!/bin/bash
# Virtual Environment Activation Script for Adobe Hackathon Project

echo "🚀 Activating Adobe Hackathon ML Environment..."

# Navigate to project root and activate virtual environment
cd "/Users/udaynarwal/Projects/Adobe Hackathon"
source venv/bin/activate

# Display environment info
echo "✅ Virtual environment activated!"
echo "🐍 Python: $(python --version)"
echo "📦 Location: $(which python)"
echo "🔥 TensorFlow: $(python -c 'import tensorflow as tf; print(tf.__version__)' 2>/dev/null || echo 'Not available')"

# Optional: navigate to workspace/app if user wants
echo ""
echo "💡 Available commands:"
echo "   cd workspace/app    # Navigate to main workspace"
echo "   cd app             # Navigate to alternative app directory"
echo "   python src/ml_pdf_extractor.py --help    # Run ML extractor"
echo ""
