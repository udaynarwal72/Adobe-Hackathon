import fitz  # PyMuPDF
import json
import os
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import argparse
from collections import defaultdict
from dataclasses import dataclass
import pickle
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # ML Libraries
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    import joblib
    
    # NLP Libraries (simple fallback if not available)
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        from nltk.stem import WordNetLemmatizer
        NLTK_AVAILABLE = True
    except ImportError:
        NLTK_AVAILABLE = False
        logger.warning("NLTK not available, using simple text processing")
    
    ML_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ML libraries not available: {e}")
    ML_AVAILABLE = False

@dataclass
class TextFeatures:
    """Class to hold text features for ML training"""
    text: str
    font_size: float
    is_bold: bool
    is_italic: bool
    is_underline: bool
    position_x: float
    position_y: float
    page_number: int
    text_length: int
    word_count: int
    has_numbers: bool
    has_punctuation: bool
    is_uppercase: bool
    is_titlecase: bool
    line_spacing: float
    font_name: str
    heading_level: Optional[str] = None

class SimplePDFOutlineExtractor:
    """Simplified ML-based PDF outline extractor without deep learning dependencies"""
    
    def __init__(self, model_path: str = "models/"):
        if not ML_AVAILABLE:
            raise ImportError("ML libraries not available. Please install scikit-learn and joblib.")
        
        self.model_path = Path(model_path)
        self.model_path.mkdir(exist_ok=True)
        
        # Initialize models
        self.heading_classifier = None
        self.tfidf_vectorizer = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Simple text processing
        if NLTK_AVAILABLE:
            try:
                nltk.data.find('tokenizers/punkt')
                nltk.data.find('corpora/stopwords')
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
            
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        else:
            self.lemmatizer = None
            self.stop_words = {'the', 'and', 'or', 'but', 'if', 'when', 'where', 'how', 'why',
                              'this', 'that', 'these', 'those', 'with', 'from', 'into', 'during'}
        
        # Load models if they exist
        self.load_models()
    
    def extract_text_features(self, doc) -> List[TextFeatures]:
        """Extract comprehensive features from PDF document"""
        features = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text_dict = page.get_text("dict")
            
            # Get page dimensions for relative positioning
            page_rect = page.rect
            page_width = page_rect.width if page_rect.width > 0 else 1
            page_height = page_rect.height if page_rect.height > 0 else 1
            
            for block in text_dict.get("blocks", []):
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        max_size = 0
                        flags = 0
                        font_name = ""
                        bbox = line.get("bbox", [0, 0, 0, 0])
                        
                        for span in line.get("spans", []):
                            line_text += span.get("text", "")
                            max_size = max(max_size, span.get("size", 12))
                            flags |= span.get("flags", 0)
                            font_name = span.get("font", "")
                        
                        if line_text.strip():
                            feature = TextFeatures(
                                text=line_text.strip(),
                                font_size=max_size,
                                is_bold=bool(flags & 2**4),
                                is_italic=bool(flags & 2**1),
                                is_underline=bool(flags & 2**0),
                                position_x=bbox[0] / page_width,
                                position_y=bbox[1] / page_height,
                                page_number=page_num + 1,
                                text_length=len(line_text.strip()),
                                word_count=len(line_text.strip().split()),
                                has_numbers=bool(re.search(r'\d', line_text)),
                                has_punctuation=bool(re.search(r'[^\w\s]', line_text)),
                                is_uppercase=line_text.isupper(),
                                is_titlecase=line_text.istitle(),
                                line_spacing=bbox[3] - bbox[1],
                                font_name=font_name
                            )
                            features.append(feature)
        
        return features
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis"""
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters but keep spaces
            text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
            
            # Simple tokenization
            tokens = text.split()
            
            # Remove stopwords and apply simple stemming
            if NLTK_AVAILABLE and self.lemmatizer:
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
            else:
                tokens = [token for token in tokens if token not in self.stop_words]
            
            return ' '.join(tokens)
        except Exception as e:
            logger.warning(f"Error preprocessing text: {e}")
            return text.lower()
    
    def extract_numerical_features(self, features: List[TextFeatures]) -> np.ndarray:
        """Extract numerical features for ML models"""
        numerical_features = []
        
        for feature in features:
            row = [
                feature.font_size,
                float(feature.is_bold),
                float(feature.is_italic),
                float(feature.is_underline),
                feature.position_x,
                feature.position_y,
                feature.page_number,
                feature.text_length,
                feature.word_count,
                float(feature.has_numbers),
                float(feature.has_punctuation),
                float(feature.is_uppercase),
                float(feature.is_titlecase),
                feature.line_spacing,
                abs(hash(feature.font_name)) % 1000  # Simple font encoding
            ]
            numerical_features.append(row)
        
        return np.array(numerical_features)
    
    def create_training_data(self, pdf_path: str, json_path: str) -> Tuple[List[TextFeatures], List[str]]:
        """Create training data from PDF and corresponding JSON output"""
        try:
            # Load PDF and extract features
            doc = fitz.open(pdf_path)
            features = self.extract_text_features(doc)
            doc.close()
            
            # Load ground truth JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                ground_truth = json.load(f)
            
            # Create labels for each text feature
            labels = []
            outline_texts = {}
            
            # Create a mapping of text to heading levels
            for item in ground_truth.get('outline', []):
                text_key = item['text'].lower().strip()
                outline_texts[text_key] = item['level']
            
            title_text = ground_truth.get('title', '').lower().strip()
            
            for feature in features:
                text_key = feature.text.lower().strip()
                
                # Check if it's the title
                if text_key == title_text or (len(title_text) > 10 and title_text in text_key):
                    labels.append('TITLE')
                # Check if it's in outline
                elif text_key in outline_texts:
                    labels.append(outline_texts[text_key])
                # Check for partial matches
                else:
                    found_match = False
                    for outline_text, level in outline_texts.items():
                        if len(outline_text) > 5 and (outline_text in text_key or text_key in outline_text):
                            labels.append(level)
                            found_match = True
                            break
                    
                    if not found_match:
                        labels.append('BODY')
            
            return features, labels
            
        except Exception as e:
            logger.error(f"Error creating training data from {pdf_path}: {e}")
            return [], []
    
    def train_models(self, training_data_dir: str):
        """Train ML models using training data"""
        logger.info("Loading training data...")
        
        all_features = []
        all_labels = []
        
        # Load all training data
        pdf_dir = Path(training_data_dir) / "pdfs"
        json_dir = Path(training_data_dir) / "outputs"
        
        if not pdf_dir.exists() or not json_dir.exists():
            raise ValueError(f"Training data directories not found. Expected: {pdf_dir} and {json_dir}")
        
        for pdf_file in pdf_dir.glob("*.pdf"):
            json_file = json_dir / f"{pdf_file.stem}.json"
            
            if json_file.exists():
                logger.info(f"Processing {pdf_file.name}...")
                features, labels = self.create_training_data(str(pdf_file), str(json_file))
                all_features.extend(features)
                all_labels.extend(labels)
        
        if not all_features:
            raise ValueError("No training data found. Please check your data directory structure.")
        
        logger.info(f"Total training samples: {len(all_features)}")
        
        # Prepare features for training
        texts = [self.preprocess_text(f.text) for f in all_features]
        numerical_features = self.extract_numerical_features(all_features)
        
        # Train TF-IDF vectorizer
        logger.info("Training TF-IDF vectorizer...")
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        text_features = self.tfidf_vectorizer.fit_transform(texts)
        
        # Combine text and numerical features
        text_features_dense = text_features.toarray()
        combined_features = np.hstack([text_features_dense, numerical_features])
        
        # Scale features
        combined_features = self.scaler.fit_transform(combined_features)
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(all_labels)
        
        # Check if we have enough samples for each class
        unique_labels, counts = np.unique(encoded_labels, return_counts=True)
        logger.info(f"Label distribution: {dict(zip(self.label_encoder.classes_[unique_labels], counts))}")
        
        # Split data
        test_size = min(0.2, 0.8)  # Use smaller test size if we have few samples
        if len(set(encoded_labels)) < 2:
            # If we only have one class, use all data for training
            X_train, X_test, y_train, y_test = combined_features, combined_features, encoded_labels, encoded_labels
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                combined_features, encoded_labels, test_size=test_size, random_state=42, stratify=encoded_labels
            )
        
        # Train Random Forest classifier
        logger.info("Training Random Forest classifier...")
        self.heading_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.heading_classifier.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.heading_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model accuracy: {accuracy:.4f}")
        
        # Print classification report
        logger.info("Classification Report:")
        try:
            logger.info(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        except Exception as e:
            logger.warning(f"Could not generate classification report: {e}")
        
        # Save models
        self.save_models()
        
        logger.info("Training completed!")
    
    def save_models(self):
        """Save trained models"""
        logger.info("Saving models...")
        
        # Save sklearn models
        joblib.dump(self.heading_classifier, self.model_path / "heading_classifier.pkl")
        joblib.dump(self.tfidf_vectorizer, self.model_path / "tfidf_vectorizer.pkl")
        joblib.dump(self.scaler, self.model_path / "scaler.pkl")
        joblib.dump(self.label_encoder, self.model_path / "label_encoder.pkl")
        
        logger.info("Models saved successfully!")
    
    def load_models(self):
        """Load trained models"""
        try:
            self.heading_classifier = joblib.load(self.model_path / "heading_classifier.pkl")
            self.tfidf_vectorizer = joblib.load(self.model_path / "tfidf_vectorizer.pkl")
            self.scaler = joblib.load(self.model_path / "scaler.pkl")
            self.label_encoder = joblib.load(self.model_path / "label_encoder.pkl")
            
            logger.info("Models loaded successfully!")
            return True
        except Exception as e:
            logger.warning(f"Could not load models: {e}")
            return False
    
    def predict_headings(self, features: List[TextFeatures]) -> List[str]:
        """Predict heading levels for text features"""
        if not self.heading_classifier or not self.tfidf_vectorizer:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Prepare features
        texts = [self.preprocess_text(f.text) for f in features]
        numerical_features = self.extract_numerical_features(features)
        
        # Transform text features
        text_features = self.tfidf_vectorizer.transform(texts)
        
        # Combine features
        text_features_dense = text_features.toarray()
        combined_features = np.hstack([text_features_dense, numerical_features])
        combined_features = self.scaler.transform(combined_features)
        
        # Predict
        predictions = self.heading_classifier.predict(combined_features)
        
        # Convert back to labels
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        return predicted_labels.tolist()
    
    def extract_title_ml(self, features: List[TextFeatures], predictions: List[str]) -> str:
        """Extract title using ML predictions"""
        title_candidates = []
        
        for feature, prediction in zip(features, predictions):
            if prediction == 'TITLE':
                title_candidates.append(feature.text)
            elif feature.page_number == 1 and feature.font_size > 14 and feature.is_bold:
                title_candidates.append(feature.text)
        
        if title_candidates:
            return title_candidates[0]
        
        # Fallback: look for large, bold text on first page
        for feature in features:
            if feature.page_number == 1 and feature.font_size > 16:
                return feature.text
        
        return "Document"
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process PDF using ML models"""
        try:
            logger.info(f"Opening PDF: {pdf_path}")
            doc = fitz.open(pdf_path)
            
            # Extract features
            logger.info("Extracting features...")
            features = self.extract_text_features(doc)
            doc.close()
            
            if not features:
                return {"title": "Document", "outline": []}
            
            # Predict headings
            logger.info("Predicting headings...")
            predictions = self.predict_headings(features)
            
            # Extract title
            title = self.extract_title_ml(features, predictions)
            
            # Create outline
            outline = []
            for feature, prediction in zip(features, predictions):
                if prediction in ['H1', 'H2', 'H3', 'H4']:
                    outline.append({
                        "level": prediction,
                        "text": feature.text,
                        "page": feature.page_number
                    })
            
            # Remove duplicates and sort
            seen = set()
            unique_outline = []
            for item in outline:
                key = (item['text'].lower().strip(), item['page'])
                if key not in seen:
                    seen.add(key)
                    unique_outline.append(item)
            
            # Sort by page and then by text
            unique_outline.sort(key=lambda x: (x['page'], x['text']))
            
            logger.info(f"Extracted {len(unique_outline)} headings")
            
            return {
                "title": title,
                "outline": unique_outline
            }
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            return {
                "title": "Error",
                "outline": []
            }

def main():
    parser = argparse.ArgumentParser(description='Simple ML-based PDF outline extraction')
    parser.add_argument('--mode', choices=['train', 'predict'], required=True, help='Mode: train or predict')
    parser.add_argument('--input', help='Input directory (for predict mode) or training data directory (for train mode)')
    parser.add_argument('--output', help='Output directory for JSON files (predict mode only)')
    parser.add_argument('--model-path', default='models/', help='Path to save/load models')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        extractor = SimplePDFOutlineExtractor(model_path=args.model_path)
        
        if args.mode == 'train':
            if not args.input:
                print("Error: --input directory required for training mode")
                return 1
            
            print("Training ML models...")
            extractor.train_models(args.input)
            
        elif args.mode == 'predict':
            if not args.input or not args.output:
                print("Error: --input and --output directories required for predict mode")
                return 1
            
            input_dir = os.path.abspath(args.input)
            output_dir = os.path.abspath(args.output)
            
            print(f"Input directory: {input_dir}")
            print(f"Output directory: {output_dir}")
            
            # Check input directory
            if not os.path.exists(input_dir):
                print(f"Error: Input directory '{input_dir}' does not exist!")
                return 1
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Find PDF files
            pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
            
            if not pdf_files:
                print(f"No PDF files found in '{input_dir}'")
                return 1
            
            print(f"Found {len(pdf_files)} PDF files to process")
            
            # Process each PDF
            for filename in pdf_files:
                pdf_path = os.path.join(input_dir, filename)
                output_filename = filename.replace('.pdf', '.json')
                output_path = os.path.join(output_dir, output_filename)
                
                print(f"\nProcessing: {filename}")
                
                result = extractor.process_pdf(pdf_path)
                
                if args.verbose:
                    print(f"Title: {result['title']}")
                    print(f"Found {len(result['outline'])} headings:")
                    for i, heading in enumerate(result['outline'][:10]):
                        print(f"  {i+1:2d}. {heading['level']}: {heading['text']} (page {heading['page']})")
                    if len(result['outline']) > 10:
                        print(f"  ... and {len(result['outline']) - 10} more")
                
                # Save result
                try:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    print(f"✓ Saved outline to: {output_path}")
                except Exception as e:
                    print(f"✗ Error saving {output_path}: {e}")
            
            print(f"\nProcessing complete! Processed {len(pdf_files)} files.")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
