# 🚀 Adobe Hackathon - ML-Powered PDF Outline Extractor

Welcome to the **Adobe Hackathon PDF Outline Extractor** repository! This project leverages machine learning to automatically extract document outlines (headings, titles, and hierarchical structure) from PDF documents with high accuracy.

## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [🔧 Quick Start](#-quick-start)
- [📁 Repository Structure](#-repository-structure)
- [🧠 ML Models & Approaches](#-ml-models--approaches)
- [💡 Key Features](#-key-features)
- [🛠️ Development Setup](#️-development-setup)
- [📖 Usage Examples](#-usage-examples)
- [🎯 Performance Metrics](#-performance-metrics)
- [🔄 Training Your Own Models](#-training-your-own-models)
- [🤝 Contributing](#-contributing)

## 🎯 Project Overview

This repository contains multiple implementations of PDF outline extraction systems, ranging from rule-based approaches to state-of-the-art machine learning models trained on ground truth data.

### What Does It Do?

- **Extracts document titles** from PDF files
- **Identifies headings** (H1, H2, H3, H4) with hierarchical structure
- **Generates structured JSON output** with page references
- **Uses multiple ML approaches** for maximum accuracy
- **Handles diverse document types** (technical papers, flyers, manuals, etc.)

### Why This Matters?

- **Document Analysis**: Automatically understand document structure
- **Content Navigation**: Create interactive tables of contents
- **Information Retrieval**: Enable better document search and indexing
- **Accessibility**: Improve document accessibility for screen readers
- **Automation**: Batch process large document collections

## 🔧 Quick Start

### Prerequisites

- Python 3.11+ (tested with Python 3.13.1)
- 4GB+ RAM (for deep learning models)
- 1GB+ storage (for models and dependencies)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/udaynarwal72/Adobe-Hackathon.git
   cd Adobe-Hackathon
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   cd app
   pip install -r requirements.txt
   ```

### Quick Test

```bash
cd app

# Process a single PDF with the enhanced ML model
python process_pdf.py input/Pdfs/E0CCG5S239.pdf

# Process all PDFs in a directory
python process_pdf.py input/Pdfs/ -o output/

# Use deep learning only (faster)
python process_pdf.py input/Pdfs/E0CCG5S239.pdf --deep-only
```

## 📁 Repository Structure

```
Adobe-Hackathon/
├── 📄 STARTER.md                    # This file - start here!
├── 📄 SUCCESS_SUMMARY.md            # Project achievements summary
├── 📁 Challenge - 1(a)/             # Ground truth training data
│   └── 📁 Datasets/
│       ├── 📁 Pdfs/                 # Training PDF files
│       └── 📁 Output.json/          # Ground truth labels
├── 📁 app/                          # Main application code
│   ├── 📁 src/                      # Source code
│   │   ├── 🐍 main.py              # Original rule-based extractor
│   │   ├── 🐍 pdf_parser.py        # PDF parsing utilities
│   │   ├── 🐍 heading_extractor.py # Rule-based heading detection
│   │   ├── 🐍 ml_pdf_extractor.py  # Machine learning extractor
│   │   └── 🐍 train_with_challenge_data.py # Advanced ML training
│   ├── 📁 input/                    # Sample input PDFs
│   │   ├── 📁 Pdfs/                # PDF samples
│   │   └── 📁 other/               # Additional samples
│   ├── 📁 output/                   # Generated JSON outputs
│   ├── 📁 models/                   # Trained ML models
│   │   ├── 🤖 rf_model_challenge.pkl     # Random Forest model
│   │   ├── 🤖 deep_model_challenge.h5    # Deep Learning model
│   │   └── 🛠️ *_challenge.pkl           # Preprocessing objects
│   ├── 🐍 enhanced_pdf_processor.py # Production-ready ML processor
│   ├── 🐍 process_pdf.py           # Simple user interface
│   ├── 🐍 train_challenge_models.py # Model training script
│   ├── 📄 requirements.txt         # Python dependencies
│   ├── 📄 Dockerfile              # Container setup
│   └── 📄 ENHANCED_ML_README.md   # Detailed ML documentation
└── 📁 appendix/                    # Additional documentation
```

## 🧠 ML Models & Approaches

### 1. **Enhanced ML System (Recommended)** 🌟
- **Location**: `enhanced_pdf_processor.py`
- **Accuracy**: 94%+ on Challenge-1(a) data
- **Models**: Random Forest + LSTM Deep Learning ensemble
- **Training Data**: 1,199 samples from ground truth
- **Features**: TF-IDF, font analysis, position detection, NLP preprocessing

### 2. **Challenge-1(a) Trainer**
- **Location**: `src/train_with_challenge_data.py`
- **Purpose**: Train models using ground truth data
- **Output**: Production-ready models with high accuracy

### 3. **General ML Extractor**
- **Location**: `src/ml_pdf_extractor.py`
- **Purpose**: Flexible ML framework for custom training
- **Features**: Supports multiple algorithms and custom datasets

### 4. **Rule-Based System**
- **Location**: `src/main.py`
- **Purpose**: Baseline implementation using heuristics
- **Accuracy**: ~75% (good for comparison)

## 💡 Key Features

### ✨ **Advanced ML Pipeline**
- **Ensemble Prediction**: Combines Random Forest + Deep Learning
- **Feature Engineering**: 15+ extracted features per text element
- **NLP Processing**: NLTK-powered text preprocessing
- **Confidence Scoring**: Know how confident each prediction is

### 🎯 **High Accuracy**
- **94%+ accuracy** on diverse document types
- **Robust handling** of various fonts and layouts
- **Hierarchical structure** preservation
- **Multi-language support** potential

### 🚀 **Production Ready**
- **Simple interfaces** for easy integration
- **Comprehensive error handling**
- **Scalable processing** for large document collections
- **Clean JSON output** format

### 🔧 **Developer Friendly**
- **Modular architecture** - easy to extend
- **Comprehensive documentation**
- **Type hints** throughout codebase
- **Professional code quality**

## 🛠️ Development Setup

### Environment Details
- **Python**: 3.13.1 (recommended)
- **TensorFlow**: 2.20.0-dev (nightly build)
- **Key Libraries**: scikit-learn, PyMuPDF, NLTK, pandas, numpy

### IDE Setup (VS Code)
1. Install Python extension
2. Set interpreter to `./venv/bin/python`
3. Enable pylint for code quality
4. Use the provided `.vscode/` settings (if available)

### Docker Setup
```bash
cd app
docker build -t pdf-extractor .
docker run -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output pdf-extractor
```

## 📖 Usage Examples

### Basic Usage
```bash
# Process single PDF
python process_pdf.py document.pdf

# Process with custom output
python process_pdf.py document.pdf -o results.json

# Process directory
python process_pdf.py pdf_folder/ -o output_folder/
```

### Advanced Usage
```bash
# Use ensemble model (default)
python enhanced_pdf_processor.py --input document.pdf --ensemble

# Use deep learning only
python enhanced_pdf_processor.py --input document.pdf --deep-only

# Process with confidence threshold
python enhanced_pdf_processor.py --input document.pdf --confidence 0.8
```

### Training Custom Models
```bash
# Train with Challenge-1(a) data
python train_challenge_models.py

# Train with custom data
python src/ml_pdf_extractor.py --mode train --input training_data/
```

## 🎯 Performance Metrics

### Model Accuracy (Challenge-1(a) Test Set)
- **Random Forest**: 93.3% accuracy
- **Deep Learning**: 94.2% accuracy
- **Ensemble**: 94%+ accuracy (best of both)

### Processing Speed
- **Rule-based**: ~2-5 seconds per PDF
- **ML-based**: ~5-15 seconds per PDF
- **Batch processing**: Scales linearly

### Supported Document Types
- ✅ Technical papers and manuals
- ✅ Academic documents
- ✅ Marketing flyers and brochures
- ✅ Legal documents
- ✅ Books and reports
- ✅ Multi-column layouts

## 🔄 Training Your Own Models

### Using Challenge-1(a) Data
```bash
cd app
python train_challenge_models.py
```

### Using Custom Dataset
1. **Prepare your data**:
   ```
   training_data/
   ├── pdfs/          # Your PDF files
   └── outputs/       # Corresponding JSON ground truth
   ```

2. **Train models**:
   ```bash
   python src/ml_pdf_extractor.py --mode train --input training_data/
   ```

3. **Test your models**:
   ```bash
   python src/ml_pdf_extractor.py --mode predict --input test_pdfs/ --output results/
   ```

### Ground Truth Format
```json
{
  "title": "Document Title",
  "outline": [
    {
      "level": "H1",
      "text": "Chapter 1: Introduction",
      "page": 1
    },
    {
      "level": "H2", 
      "text": "1.1 Overview",
      "page": 1
    }
  ]
}
```

## 🤝 Contributing

### Getting Started
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Ensure code quality: `python -m pylint src/`
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Development Guidelines
- **Code Quality**: Follow PEP 8 style guidelines
- **Type Hints**: Add type hints to all functions
- **Documentation**: Document all public methods
- **Testing**: Add tests for new features
- **Performance**: Consider computational efficiency

### Areas for Contribution
- 🔍 **New ML Models**: Experiment with different algorithms
- 🌐 **Multi-language Support**: Add support for non-English documents  
- ⚡ **Performance Optimization**: Speed up processing
- 📊 **Evaluation Metrics**: Add more comprehensive evaluation
- 🎨 **UI/UX**: Create web interface or GUI
- 📚 **Documentation**: Improve guides and tutorials

## 🏆 Project Achievements

### What We've Built
- ✅ **State-of-the-art ML pipeline** with 94%+ accuracy
- ✅ **Production-ready system** with clean interfaces
- ✅ **Comprehensive training pipeline** using ground truth data
- ✅ **Multiple implementation approaches** for different use cases
- ✅ **Professional documentation** and code quality

### Key Innovations
- **Ensemble ML approach** combining traditional and deep learning
- **Advanced feature engineering** with 15+ text properties
- **Ground truth training** using Challenge-1(a) dataset
- **Scalable architecture** for batch processing
- **Developer-friendly interfaces** for easy adoption

## 📞 Support & Contact

### Getting Help
- 📖 **Documentation**: Check `ENHANCED_ML_README.md` for detailed ML docs
- 🐛 **Issues**: Open GitHub issues for bugs or feature requests
- 💬 **Discussions**: Use GitHub discussions for questions

### Project Maintainer
- **GitHub**: [@udaynarwal72](https://github.com/udaynarwal72)
- **Repository**: [Adobe-Hackathon](https://github.com/udaynarwal72/Adobe-Hackathon)

---

## 🚀 Ready to Start?

1. **New to the project?** Start with the [Quick Start](#-quick-start) section
2. **Want to use ML models?** Check out `process_pdf.py` for the simplest interface
3. **Need to train custom models?** See the [Training section](#-training-your-own-models)
4. **Contributing?** Read the [Contributing guidelines](#-contributing)
5. **Looking for details?** Dive into `ENHANCED_ML_README.md`

**Happy coding! 🎉**

---

*This project was developed for the Adobe Hackathon to demonstrate advanced PDF processing capabilities using machine learning. The goal is to make document structure extraction accessible, accurate, and production-ready.*
