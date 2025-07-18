#!/usr/bin/env python3
"""
Enhanced PDF Processor using Challenge-1(a) trained models
Processes PDFs with the newly trained ML models for better accuracy
Form-aware version that can handle numbered items and form fields
REFINED VERSION - Better extraction and filtering
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
                
            print("[SUCCESS] Models loaded successfully!")
            print(f"Available classes: {list(self.label_encoder.classes_)}")
            
        except Exception as e:
            print(f"[ERROR] Error loading models: {e}")
            raise
    
    def extract_pdf_features(self, pdf_path: str) -> List[Dict]:
        """Extract features from PDF with better spatial understanding"""
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
                                if text and len(text) > 0:  # Keep even single characters for better context
                                    
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
                                        'height': height,
                                        'line_index': len(features)  # Track order
                                    }
                                    features.append(feature)
            
            doc.close()
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            
        return features
    
    def is_form_number(self, text: str) -> bool:
        """Check if text is just a form number (like '1.', '2.', etc.)"""
        # Exact match for numbered items
        if re.match(r'^\d+\.\s*$', text.strip()):
            return True
        
        # Exact match for lettered items
        if re.match(r'^[a-zA-Z]\.\s*$', text.strip()):
            return True
            
        # Exact match for roman numerals
        if re.match(r'^[ivxlcdm]+\.\s*$', text.strip().lower()):
            return True
        
        # Exact match for parenthetical numbers
        if re.match(r'^\(\d+\)\s*$', text.strip()):
            return True
            
        return False
    
    def is_meaningful_heading(self, text: str) -> bool:
        """Check if text contains meaningful content for a heading"""
        text = text.strip()
        
        # Must have reasonable length
        if len(text) < 3:
            return False
            
        # Skip standalone numbers or letters
        if re.match(r'^\d+\.?\s*$', text) or re.match(r'^[a-zA-Z]\.?\s*$', text):
            return False
        
        # Must contain at least one meaningful word
        meaningful_words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        if not meaningful_words:
            return False
        
        # Check for form-related keywords
        form_keywords = [
            'name', 'designation', 'date', 'service', 'pay', 'salary', 'permanent', 
            'temporary', 'home', 'town', 'wife', 'husband', 'employed', 'entitled',
            'concession', 'availed', 'visiting', 'block', 'place', 'visit', 'india',
            'fare', 'rail', 'bus', 'headquarters', 'route', 'persons', 'respect',
            'relationship', 'amount', 'advance', 'required', 'declare', 'particulars',
            'furnished', 'true', 'correct', 'knowledge', 'undertake', 'produce',
            'tickets', 'journey', 'receipt', 'cancellation', 'refund', 'signature',
            'government', 'servant', 'application', 'form', 'grant'
        ]
        
        # Higher score for texts with form keywords
        keyword_score = sum(1 for word in meaningful_words if word in form_keywords)
        
        # Must have at least one keyword or be substantial text
        return keyword_score > 0 or len(text) > 15
    
    def find_associated_text(self, features: List[Dict], start_index: int, max_distance: int = 10) -> str:
        """Find text associated with a form number"""
        if start_index >= len(features):
            return ""
        
        form_number = features[start_index]['text'].strip()
        associated_texts = [form_number]
        
        current_y = features[start_index]['y_position']
        
        # Look forward for associated text
        for i in range(start_index + 1, min(start_index + max_distance, len(features))):
            feature = features[i]
            text = feature['text'].strip()
            
            # Stop if we hit another form number
            if self.is_form_number(text):
                break
            
            # Calculate distance from the form number
            y_distance = abs(feature['y_position'] - current_y)
            
            # Include text if it's close enough and meaningful
            if y_distance < 30 and len(text) > 0:  # Within 30 pixels
                associated_texts.append(text)
                current_y = feature['y_position']  # Update current position
                
                # Stop if we have enough content
                if len(' '.join(associated_texts)) > 50:
                    break
            elif y_distance > 50:  # Too far away
                break
        
        return ' '.join(associated_texts)
    
    def extract_form_structure(self, features: List[Dict]) -> List[Dict]:
        """Extract form structure with better text association"""
        form_headings = []
        processed_indices = set()
        
        # First, find and add the main title
        title_candidates = []
        for i, feature in enumerate(features):
            text = feature['text'].strip()
            
            # Look for title-like text
            if (len(text) > 20 and 
                feature['page'] == 1 and 
                feature['y_position'] < 150 and  # Near top of page
                not self.is_form_number(text) and
                ('application' in text.lower() or 'form' in text.lower() or 
                 'grant' in text.lower() or 'advance' in text.lower())):
                title_candidates.append({
                    'level': 'H1',
                    'text': text,
                    'page': feature['page'],
                    'priority': 1
                })
                processed_indices.add(i)
        
        # Add best title candidate
        if title_candidates:
            # Sort by length (longer titles are usually better)
            title_candidates.sort(key=lambda x: len(x['text']), reverse=True)
            form_headings.append(title_candidates[0])
        
        # Process numbered items with their associated text
        for i, feature in enumerate(features):
            if i in processed_indices:
                continue
                
            text = feature['text'].strip()
            
            # Check if it's a form number
            if self.is_form_number(text):
                # Find associated text
                full_text = self.find_associated_text(features, i)
                
                # Only include if the associated text is meaningful
                if self.is_meaningful_heading(full_text):
                    # Determine heading level based on the number
                    if re.match(r'^\d+\.\s', full_text):
                        level = 'H2'
                    elif re.match(r'^[a-zA-Z]\.\s', full_text):
                        level = 'H3'
                    else:
                        level = 'H3'
                    
                    form_headings.append({
                        'level': level,
                        'text': full_text.strip(),
                        'page': feature['page'],
                        'priority': 2
                    })
                
                # Mark associated features as processed
                for j in range(i, min(i + 10, len(features))):
                    if features[j]['y_position'] - feature['y_position'] < 30:
                        processed_indices.add(j)
        
        # Look for standalone important phrases
        for i, feature in enumerate(features):
            if i in processed_indices:
                continue
                
            text = feature['text'].strip()
            
            # Check for important standalone phrases
            if (len(text) > 10 and 
                self.is_meaningful_heading(text) and
                not self.is_form_number(text)):
                
                # Check if it's not just part of a sentence
                if (not text.endswith(',') and 
                    not text.startswith('and') and
                    not text.startswith('or') and
                    not text.startswith('the')):
                    
                    form_headings.append({
                        'level': 'H3',
                        'text': text,
                        'page': feature['page'],
                        'priority': 3
                    })
        
        # Sort by priority and page order
        form_headings.sort(key=lambda x: (x['priority'], x['page']))
        
        # Remove duplicates and clean up
        seen_texts = set()
        unique_headings = []
        
        for heading in form_headings:
            # Clean up text
            clean_text = re.sub(r'\s+', ' ', heading['text']).strip()
            
            # Skip if we've seen similar text
            if clean_text.lower() not in seen_texts and len(clean_text) > 2:
                seen_texts.add(clean_text.lower())
                unique_headings.append({
                    'level': heading['level'],
                    'text': clean_text,
                    'page': heading['page']
                })
        
        return unique_headings
    
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
    
    def post_process_headings(self, headings: List[Dict]) -> List[Dict]:
        """Post-process headings to improve quality"""
        if not headings:
            return []
        
        processed_headings = []
        
        for heading in headings:
            text = heading['text'].strip()
            
            # Skip very short or meaningless headings
            if len(text) < 3:
                continue
            
            # Skip standalone numbers or letters
            if re.match(r'^\d+\.?\s*$', text) or re.match(r'^[a-zA-Z]\.?\s*$', text):
                continue
            
            # Skip if it's just punctuation
            if re.match(r'^[^\w\s]+$', text):
                continue
            
            # Clean up the text
            clean_text = re.sub(r'\s+', ' ', text).strip()
            
            # Ensure minimum meaningful content
            if self.is_meaningful_heading(clean_text):
                processed_headings.append({
                    'level': heading['level'],
                    'text': clean_text,
                    'page': heading['page']
                })
        
        return processed_headings
    
    def predict_headings(self, features: List[Dict], use_ensemble: bool = True, use_form_fallback: bool = True) -> List[Dict]:
        """Predict heading levels using trained models with form fallback"""
        if not features:
            return []
        
        # Try ML models first
        ml_results = []
        
        try:
            # Prepare features
            X_combined, X_text, X_numerical_dl = self.prepare_features_for_prediction(features)
            
            # Get predictions from both models
            rf_predictions = self.rf_model.predict(X_combined)
            dl_predictions = self.deep_model.predict([X_text, X_numerical_dl])
            dl_predictions = np.argmax(dl_predictions, axis=1)
            
            # Convert predictions to labels
            rf_labels = self.label_encoder.inverse_transform(rf_predictions)
            dl_labels = self.label_encoder.inverse_transform(dl_predictions)
            
            # Ensemble prediction
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
            
            # Create ML results
            for i, feature in enumerate(features):
                if final_predictions[i] != 'TEXT':  # Only include headings
                    ml_results.append({
                        'level': final_predictions[i],
                        'text': feature['text'],
                        'page': feature['page']
                    })
            
            # Post-process ML results
            ml_results = self.post_process_headings(ml_results)
            
        except Exception as e:
            print(f"[WARNING] ML prediction failed: {e}")
            ml_results = []
        
        # If ML models found good headings, use them
        if len(ml_results) > 3:  # Only use ML if we found a reasonable number of headings
            print(f"[INFO] ML models found {len(ml_results)} headings")
            return ml_results
        
        # Fallback to form structure extraction
        if use_form_fallback:
            print("[INFO] Using form structure extraction as fallback")
            form_results = self.extract_form_structure(features)
            print(f"[INFO] Form extraction found {len(form_results)} headings")
            return form_results
        
        return ml_results if ml_results else []

    # ...existing code...
    def process_pdf(self, pdf_path: str, output_path: Optional[str] = None, use_ensemble: bool = True, use_form_fallback: bool = True) -> Dict:
        print(f"Processing: {pdf_path}")

        # Extract features
        features = self.extract_pdf_features(pdf_path)
        print(f"Extracted {len(features)} text elements")

        # Predict headings
        headings = self.predict_headings(features, use_ensemble=use_ensemble, use_form_fallback=use_form_fallback)
        print(f"Identified {len(headings)} headings")

        # Try to use first H1 or H2 heading as title
        title = None
        for heading in headings:
            if heading['level'] == 'H1':
                title = heading['text']
                break
        if not title:
            for heading in headings:
                if heading['level'] == 'H2':
                    title = heading['text']
                    break
        if not title:
            title = os.path.basename(pdf_path).replace('.pdf', '')

        # Create output structure
        output = {
            'title': title,
            'outline': headings
        }

        # Save output if path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {output_path}")

        return output
# ...existing code...

    def process_directory(self, input_dir: str, output_dir: str, use_ensemble: bool = True, use_form_fallback: bool = True):
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
                result = self.process_pdf(pdf_path, output_path, use_ensemble, use_form_fallback)
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
                print(f"[SUCCESS] {result['file']}: {result['headings_found']} headings")
            else:
                print(f"[ERROR] {result['file']}: {result['error']}")
        
        return results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Process PDFs with Challenge-1(a) trained models")
    parser.add_argument("--input", required=True, help="Input PDF file or directory")
    parser.add_argument("--output", help="Output JSON file or directory")
    parser.add_argument("--models-path", default="models", help="Path to models directory")
    parser.add_argument("--ensemble", action="store_true", default=True, help="Use ensemble prediction")
    parser.add_argument("--deep-only", action="store_true", help="Use deep learning model only")
    parser.add_argument("--no-form-fallback", action="store_true", help="Disable form structure fallback")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = EnhancedPDFProcessor(args.models_path)
    
    use_ensemble = args.ensemble and not args.deep_only
    use_form_fallback = not args.no_form_fallback
    
    if os.path.isfile(args.input):
        # Process single file
        result = processor.process_pdf(args.input, args.output, use_ensemble, use_form_fallback)
        print(f"\nProcessed {args.input}:")
        print(f"Found {len(result['outline'])} headings")
        
        # Print sample output format
        if result['outline']:
            print("\nSample output format:")
            sample_output = {
                'title': result['title'],
                'outline': result['outline'][:5]  # Show first 5 headings
            }
            print(json.dumps(sample_output, indent=2))
        
    elif os.path.isdir(args.input):
        # Process directory
        if not args.output:
            args.output = "output_enhanced"
        processor.process_directory(args.input, args.output, use_ensemble, use_form_fallback)
    
    else:
        print(f"Error: {args.input} is not a valid file or directory")

if __name__ == "__main__":
    main()