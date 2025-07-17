#!/usr/bin/env python3
"""
Enhanced PDF Processor using Challenge-1(a) trained models
Processes PDFs with the newly trained ML models for better accuracy
"""

import fitz  # PyMuPDF
import json
import os
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import argparse
import pickle
from pathlib import Path

# ML Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class EnhancedPDFProcessor:
    """Enhanced PDF processor using Challenge-1(a) trained models"""
    
    def __init__(self, models_path: str = "models"):
        """Initialize processor with trained models"""
        self.models_path = models_path
        
        # Initialize preprocessing tools
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Load models and preprocessors
        self.load_models()
        
    def load_models(self):
        """Load trained models and preprocessors"""
        try:
            # Load Random Forest model
            self.rf_model = joblib.load(os.path.join(self.models_path, 'rf_model_challenge.pkl'))
            
            # Load Deep Learning model
            self.deep_model = load_model(os.path.join(self.models_path, 'deep_model_challenge.h5'))
            
            # Load preprocessors
            self.vectorizer = joblib.load(os.path.join(self.models_path, 'vectorizer_challenge.pkl'))
            self.scaler = joblib.load(os.path.join(self.models_path, 'scaler_challenge.pkl'))
            self.label_encoder = joblib.load(os.path.join(self.models_path, 'label_encoder_challenge.pkl'))
            
            # Load tokenizer
            with open(os.path.join(self.models_path, 'tokenizer_challenge.pkl'), 'rb') as f:
                self.tokenizer = pickle.load(f)
                
            print("✅ Models loaded successfully!")
            print(f"Available classes: {list(self.label_encoder.classes_)}")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def extract_pdf_features(self, pdf_path: str) -> List[Dict]:
        """Extract features from PDF"""
        features = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks = page.get_text("dict")  # type: ignore
                
                for block in blocks.get("blocks", []):
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text = span["text"].strip()
                                if text and len(text) > 2:  # Filter very short texts
                                    
                                    # Extract font information
                                    font_size = span.get("size", 12)
                                    font_name = span.get("font", "")
                                    flags = span.get("flags", 0)
                                    
                                    # Determine font properties
                                    is_bold = bool(flags & 2**4)
                                    is_italic = bool(flags & 2**1)
                                    
                                    # Extract position information
                                    bbox = span.get("bbox", [0, 0, 0, 0])
                                    x_pos = bbox[0]
                                    y_pos = bbox[1]
                                    width = bbox[2] - bbox[0]
                                    height = bbox[3] - bbox[1]
                                    
                                    feature = {
                                        'text': text,
                                        'page': page_num + 1,
                                        'font_size': font_size,
                                        'font_name': font_name,
                                        'is_bold': is_bold,
                                        'is_italic': is_italic,
                                        'x_position': x_pos,
                                        'y_position': y_pos,
                                        'width': width,
                                        'height': height
                                    }
                                    features.append(feature)
            
            doc.close()
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            
        return features
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for feature extraction"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def prepare_features_for_prediction(self, features: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare features for model prediction"""
        # Extract text features
        texts = [f['text'] for f in features]
        
        # Preprocess text
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Create TF-IDF features
        text_features = self.vectorizer.transform(processed_texts).toarray()
        
        # Extract numerical features
        numerical_features = []
        for f in features:
            num_feat = [
                f['font_size'],
                len(f['text']),
                f['x_position'],
                f['y_position'],
                f['width'],
                f['height'],
                int(f['is_bold']),
                int(f['is_italic']),
                f['page']
            ]
            numerical_features.append(num_feat)
        
        numerical_features = np.array(numerical_features)
        
        # Scale numerical features
        numerical_features = self.scaler.transform(numerical_features)
        
        # Combine features for Random Forest
        X_combined = np.hstack([text_features, numerical_features])
        
        # Prepare text sequences for deep learning
        sequences = self.tokenizer.texts_to_sequences(texts)
        X_text = pad_sequences(sequences, maxlen=50, padding='post')
        
        # Normalize numerical features for deep learning
        X_numerical_dl = []
        for f in features:
            num_feat = [
                f['font_size'] / 20.0,  # Normalize font size
                len(f['text']) / 100.0,  # Normalize text length
                f['x_position'] / 1000.0,  # Normalize position
                f['y_position'] / 1000.0,
                f['width'] / 1000.0,
                f['height'] / 100.0,
                float(f['is_bold']),
                float(f['is_italic']),
                f['page'] / 10.0  # Normalize page number
            ]
            X_numerical_dl.append(num_feat)
        
        X_numerical_dl = np.array(X_numerical_dl)
        
        return X_combined, X_text, X_numerical_dl
    
    def predict_headings(self, features: List[Dict], use_ensemble: bool = True) -> List[Dict]:
        """Predict heading levels using trained models"""
        if not features:
            return []
        
        # Prepare features
        X_combined, X_text, X_numerical_dl = self.prepare_features_for_prediction(features)
        
        # Get predictions from both models
        rf_predictions = self.rf_model.predict(X_combined)
        dl_predictions = self.deep_model.predict([X_text, X_numerical_dl])
        dl_predictions = np.argmax(dl_predictions, axis=1)
        
        # Convert predictions to labels
        rf_labels = self.label_encoder.inverse_transform(rf_predictions)
        dl_labels = self.label_encoder.inverse_transform(dl_predictions)
        
        # Ensemble prediction (you can customize this logic)
        final_predictions = []
        for i, (rf_pred, dl_pred) in enumerate(zip(rf_labels, dl_labels)):
            if use_ensemble:
                # Use deep learning prediction if it's a heading, otherwise use random forest
                if dl_pred != 'TEXT':
                    final_pred = dl_pred
                else:
                    final_pred = rf_pred
            else:
                # Use deep learning model by default
                final_pred = dl_pred
            
            final_predictions.append(final_pred)
        
        # Create results
        results = []
        for i, feature in enumerate(features):
            if final_predictions[i] != 'TEXT':  # Only include headings
                result = {
                    'level': final_predictions[i],
                    'text': feature['text'],
                    'page': feature['page'],
                    'confidence': {
                        'rf_prediction': rf_labels[i],
                        'dl_prediction': dl_labels[i],
                        'final_prediction': final_predictions[i]
                    }
                }
                results.append(result)
        
        return results
    
    def process_pdf(self, pdf_path: str, output_path: Optional[str] = None, use_ensemble: bool = True) -> Dict:
        """Process a PDF and extract headings"""
        print(f"Processing: {pdf_path}")
        
        # Extract features
        features = self.extract_pdf_features(pdf_path)
        print(f"Extracted {len(features)} text elements")
        
        # Predict headings
        headings = self.predict_headings(features, use_ensemble=use_ensemble)
        print(f"Identified {len(headings)} headings")
        
        # Create output structure
        pdf_name = os.path.basename(pdf_path).replace('.pdf', '')
        output = {
            'title': pdf_name,
            'outline': headings,
            'metadata': {
                'total_text_elements': len(features),
                'total_headings': len(headings),
                'model_used': 'Challenge-1(a) trained ensemble' if use_ensemble else 'Deep Learning only',
                'heading_distribution': {}
            }
        }
        
        # Calculate heading distribution
        for heading in headings:
            level = heading['level']
            if level in output['metadata']['heading_distribution']:
                output['metadata']['heading_distribution'][level] += 1
            else:
                output['metadata']['heading_distribution'][level] = 1
        
        # Save output if path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {output_path}")
        
        return output
    
    def process_directory(self, input_dir: str, output_dir: str, use_ensemble: bool = True):
        """Process all PDFs in a directory"""
        os.makedirs(output_dir, exist_ok=True)
        
        pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
        
        print(f"Found {len(pdf_files)} PDF files to process")
        
        results = []
        for pdf_file in pdf_files:
            pdf_path = os.path.join(input_dir, pdf_file)
            output_file = pdf_file.replace('.pdf', '.json')
            output_path = os.path.join(output_dir, output_file)
            
            try:
                result = self.process_pdf(pdf_path, output_path, use_ensemble)
                results.append({
                    'file': pdf_file,
                    'status': 'success',
                    'headings_found': len(result['outline'])
                })
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
                results.append({
                    'file': pdf_file,
                    'status': 'error',
                    'error': str(e)
                })
        
        print("\n" + "="*50)
        print("PROCESSING COMPLETE!")
        print("="*50)
        for result in results:
            if result['status'] == 'success':
                print(f"✅ {result['file']}: {result['headings_found']} headings")
            else:
                print(f"❌ {result['file']}: {result['error']}")
        
        return results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Process PDFs with Challenge-1(a) trained models")
    parser.add_argument("--input", required=True, help="Input PDF file or directory")
    parser.add_argument("--output", help="Output JSON file or directory")
    parser.add_argument("--models-path", default="models", help="Path to models directory")
    parser.add_argument("--ensemble", action="store_true", default=True, help="Use ensemble prediction")
    parser.add_argument("--deep-only", action="store_true", help="Use deep learning model only")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = EnhancedPDFProcessor(args.models_path)
    
    use_ensemble = args.ensemble and not args.deep_only
    
    if os.path.isfile(args.input):
        # Process single file
        result = processor.process_pdf(args.input, args.output, use_ensemble)
        print(f"\nProcessed {args.input}:")
        print(f"Found {len(result['outline'])} headings")
        
    elif os.path.isdir(args.input):
        # Process directory
        if not args.output:
            args.output = "output_enhanced"
        processor.process_directory(args.input, args.output, use_ensemble)
    
    else:
        print(f"Error: {args.input} is not a valid file or directory")

if __name__ == "__main__":
    main()
