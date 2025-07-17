# ML PDF Outline Extractor

This is a machine learning-based PDF outline extraction system that can be trained on your specific datasets to achieve high accuracy in identifying document titles and heading hierarchies.

## Features

- **Machine Learning-based**: Uses Random Forest classifier trained on your specific data
- **Comprehensive Feature Extraction**: Analyzes font size, formatting, positioning, and text content
- **Trainable**: Can be trained on your PDF-JSON pairs for improved accuracy
- **Robust**: Handles various PDF formats and layouts

## Setup

1. **Install Dependencies**:
   ```bash
   chmod +x setup_ml.sh
   ./setup_ml.sh
   ```

2. **Activate Virtual Environment**:
   ```bash
   source venv/bin/activate
   ```

## Usage

### 1. Prepare Training Data

First, organize your training data using the data preparation script:

```bash
python src/prepare_data.py --source /path/to/your/data --target training_data
```

This will organize your PDFs and JSON files into the correct structure:
```
training_data/
├── pdfs/
│   ├── document1.pdf
│   ├── document2.pdf
│   └── ...
└── outputs/
    ├── document1.json
    ├── document2.json
    └── ...
```

### 2. Train the Model

Train the ML model on your data:

```bash
python src/ml_pdf_extractor_v2.py --mode train --input training_data --verbose
```

### 3. Use the Trained Model

Use the trained model to extract outlines from new PDFs:

```bash
python src/ml_pdf_extractor_v2.py --mode predict --input input/Pdfs --output output --verbose
```

## Training Data Format

Your JSON training files should follow this format:

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
            "page": 2
        },
        {
            "level": "H3",
            "text": "Key Points",
            "page": 3
        }
    ]
}
```

## Model Features

The ML model analyzes multiple features:

- **Text Features**: Content analysis using TF-IDF
- **Font Features**: Size, bold, italic, underline formatting
- **Position Features**: X/Y coordinates, page number
- **Content Features**: Length, word count, numbers, punctuation
- **Style Features**: Upper/title case, font name

## File Structure

```
app/
├── src/
│   ├── ml_pdf_extractor_v2.py    # Main ML extractor
│   ├── prepare_data.py           # Data preparation utility
│   └── main.py                   # Original rule-based extractor
├── models/                       # Trained models will be saved here
├── training_data/               # Your training data
│   ├── pdfs/
│   └── outputs/
├── requirements_ml.txt          # Python dependencies
└── setup_ml.sh                 # Setup script
```

## Example Workflow

1. **Prepare your data**:
   ```bash
   # Put your PDFs in one directory and JSON files in another
   python src/prepare_data.py --source /path/to/your/data --target training_data
   ```

2. **Train the model**:
   ```bash
   python src/ml_pdf_extractor_v2.py --mode train --input training_data --verbose
   ```

3. **Test on new PDFs**:
   ```bash
   python src/ml_pdf_extractor_v2.py --mode predict --input input/Pdfs --output output --verbose
   ```

## Troubleshooting

- **Import errors**: Make sure you've activated the virtual environment and installed dependencies
- **No training data**: Check that your PDF and JSON files have matching names
- **Low accuracy**: Add more training examples or verify your JSON format
- **Memory issues**: Use smaller batch sizes or reduce feature dimensions

## Performance Tips

- Use at least 20-30 training examples for good results
- Ensure your training data covers various document types
- Validate your JSON format using the data preparation script
- Use the verbose flag to monitor training progress

## Dependencies

- PyMuPDF (fitz) - PDF processing
- scikit-learn - Machine learning
- nltk - Natural language processing  
- numpy, pandas - Data processing
- joblib - Model persistence
