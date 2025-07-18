#!/usr/bin/env python3
"""
Enhanced ML PDF Extractor with Challenge-1(a) Training Data
Uses the ground truth data from Challenge-1(a) to train improved models
"""

import fitz  # PyMuPDF
import json
import os
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import argparse
from collections import defaultdict
from dataclasses import dataclass
import pickle
from pathlib import Path
import logging

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model  # type: ignore
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, BatchNormalization, Input, Concatenate  # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

@dataclass
class TextFeatures:
    """Class to hold text features for ML training"""
    text: str
    page: int
    font_size: float
    font_name: str
    is_bold: bool
    is_italic: bool
    x_position: float
    y_position: float
    width: float
    height: float
    level: str  # Ground truth level (H1, H2, H3)

class ChallengeDataTrainer:
    """Enhanced trainer using Challenge-1(a) ground truth data"""
    
    def __init__(self, challenge_data_path: str, app_data_path: str):
        """
        Initialize trainer with paths to challenge data and app data
        
        Args:
            challenge_data_path: Path to Challenge-1(a) folder
            app_data_path: Path to app folder
        """
        self.challenge_pdfs_path = os.path.join(challenge_data_path, "Datasets", "Pdfs")
        self.challenge_json_path = os.path.join(challenge_data_path, "Datasets", "Output.json")
        self.app_data_path = app_data_path
        self.model_path = os.path.join(app_data_path, "models")
        
        # Create models directory if it doesn't exist
        os.makedirs(self.model_path, exist_ok=True)
        
        # Initialize preprocessing tools
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize models
        self.rf_model = None
        self.deep_model = None
        self.tokenizer = None
        
    def load_ground_truth_data(self) -> List[TextFeatures]:
        """Load ground truth data from Challenge-1(a) datasets"""
        logger.info("Loading ground truth data from Challenge-1(a)...")
        
        all_features = []
        
        # Get list of PDF files
        pdf_files = [f for f in os.listdir(self.challenge_pdfs_path) if f.endswith('.pdf')]
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.challenge_pdfs_path, pdf_file)
            json_file = pdf_file.replace('.pdf', '.json')
            json_path = os.path.join(self.challenge_json_path, json_file)
            
            if not os.path.exists(json_path):
                logger.warning(f"Ground truth file not found for {pdf_file}")
                continue
                
            logger.info(f"Processing {pdf_file}...")
            
            # Load ground truth
            with open(json_path, 'r', encoding='utf-8') as f:
                ground_truth = json.load(f)
            
            # Create mapping of text to level
            text_to_level = {}
            for item in ground_truth.get('outline', []):
                text_to_level[item['text'].strip()] = item['level']
            
            # Extract PDF features
            pdf_features = self.extract_pdf_features(pdf_path)
            
            # Match with ground truth
            for feature in pdf_features:
                text_clean = feature.text.strip()
                if text_clean in text_to_level:
                    feature.level = text_to_level[text_clean]
                    all_features.append(feature)
                else:
                    # If exact match not found, try fuzzy matching
                    best_match = self.find_best_match(text_clean, list(text_to_level.keys()))
                    if best_match and self.similarity_ratio(text_clean, best_match) > 0.8:
                        feature.level = text_to_level[best_match]
                        all_features.append(feature)
                    else:
                        # Default to regular text if no match found
                        feature.level = "TEXT"
                        all_features.append(feature)
        
        logger.info(f"Loaded {len(all_features)} training samples")
        return all_features
    
    def find_best_match(self, text: str, candidates: List[str]) -> Optional[str]:
        """Find best matching text from candidates"""
        from difflib import SequenceMatcher
        
        best_match = None
        best_ratio = 0
        
        for candidate in candidates:
            ratio = SequenceMatcher(None, text.lower(), candidate.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = candidate
                
        return best_match if best_ratio > 0.7 else None
    
    def similarity_ratio(self, text1: str, text2: str) -> float:
        """Calculate similarity ratio between two texts"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def extract_pdf_features(self, pdf_path: str) -> List[TextFeatures]:
        """Extract features from PDF for training"""
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
                                    
                                    feature = TextFeatures(
                                        text=text,
                                        page=page_num + 1,
                                        font_size=font_size,
                                        font_name=font_name,
                                        is_bold=is_bold,
                                        is_italic=is_italic,
                                        x_position=x_pos,
                                        y_position=y_pos,
                                        width=width,
                                        height=height,
                                        level="TEXT"  # Will be updated with ground truth
                                    )
                                    features.append(feature)
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            
        return features
    
    def prepare_training_data(self, features: List[TextFeatures]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels for training"""
        logger.info("Preparing training data...")
        
        # Extract text features
        texts = [f.text for f in features]
        
        # Preprocess text
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Create TF-IDF features
        text_features = self.vectorizer.fit_transform(processed_texts).toarray()  # type: ignore
        
        # Extract numerical features
        numerical_features = []
        for f in features:
            num_feat = [
                f.font_size,
                len(f.text),
                f.x_position,
                f.y_position,
                f.width,
                f.height,
                int(f.is_bold),
                int(f.is_italic),
                f.page
            ]
            numerical_features.append(num_feat)
        
        numerical_features = np.array(numerical_features)
        
        # Scale numerical features
        numerical_features = self.scaler.fit_transform(numerical_features)
        
        # Combine features
        X = np.hstack([text_features, numerical_features])
        
        # Prepare labels
        labels = [f.level for f in features]
        y = self.label_encoder.fit_transform(labels)
        
        logger.info(f"Training data shape: {X.shape}")
        logger.info(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
        
        return X, np.array(y)  # Convert y to numpy array
    
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
    
    def train_random_forest(self, X: np.ndarray, y: np.ndarray) -> float:
        """Train Random Forest model"""
        logger.info("Training Random Forest model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train model
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.rf_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Random Forest Accuracy: {accuracy:.4f}")
        logger.info("Classification Report:")
        logger.info(classification_report(y_test, y_pred, 
                                        target_names=self.label_encoder.classes_))
        
        # Save model
        joblib.dump(self.rf_model, os.path.join(self.model_path, 'rf_model_challenge.pkl'))
        
        return float(accuracy)  # Convert to Python float
    
    def prepare_deep_learning_data(self, features: List[TextFeatures]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for deep learning model"""
        logger.info("Preparing deep learning data...")
        
        # Prepare text sequences
        texts = [f.text for f in features]
        
        # Initialize and fit tokenizer
        self.tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(texts)
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # Pad sequences
        max_length = 50
        X_text = pad_sequences(sequences, maxlen=max_length, padding='post')
        
        # Prepare numerical features
        numerical_features = []
        for f in features:
            num_feat = [
                f.font_size / 20.0,  # Normalize font size
                len(f.text) / 100.0,  # Normalize text length
                f.x_position / 1000.0,  # Normalize position
                f.y_position / 1000.0,
                f.width / 1000.0,
                f.height / 100.0,
                float(f.is_bold),
                float(f.is_italic),
                f.page / 10.0  # Normalize page number
            ]
            numerical_features.append(num_feat)
        
        X_numerical = np.array(numerical_features)
        
        # Prepare labels
        labels = [f.level for f in features]
        y_encoded = self.label_encoder.transform(labels)
        y_categorical = to_categorical(y_encoded)
        
        return X_text, X_numerical, y_categorical
    
    def train_deep_learning_model(self, X_text: np.ndarray, X_numerical: np.ndarray, y: np.ndarray) -> float:
        """Train deep learning model"""
        logger.info("Training deep learning model...")
        
        # Split data
        indices = np.arange(len(X_text))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
        
        X_text_train, X_text_test = X_text[train_idx], X_text[test_idx]
        X_num_train, X_num_test = X_numerical[train_idx], X_numerical[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Build model
        # Text input branch
        text_input = Input(shape=(X_text.shape[1],), name='text_input')
        embedding = Embedding(input_dim=5000, output_dim=128, input_length=X_text.shape[1])(text_input)
        lstm = LSTM(64, dropout=0.3, recurrent_dropout=0.3)(embedding)
        
        # Numerical input branch
        num_input = Input(shape=(X_numerical.shape[1],), name='numerical_input')
        dense_num = Dense(32, activation='relu')(num_input)
        dense_num = Dropout(0.3)(dense_num)
        
        # Combine branches
        combined = Concatenate()([lstm, dense_num])
        dense_combined = Dense(64, activation='relu')(combined)
        dense_combined = BatchNormalization()(dense_combined)
        dense_combined = Dropout(0.5)(dense_combined)
        
        output = Dense(y.shape[1], activation='softmax')(dense_combined)
        
        # Create and compile model
        self.deep_model = Model(inputs=[text_input, num_input], outputs=output)
        self.deep_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = self.deep_model.fit(
            [X_text_train, X_num_train], y_train,
            validation_data=([X_text_test, X_num_test], y_test),
            epochs=50,
            batch_size=32,
            verbose=1
        )
        
        # Evaluate
        loss, accuracy = self.deep_model.evaluate([X_text_test, X_num_test], y_test, verbose=0)
        
        logger.info(f"Deep Learning Model Accuracy: {accuracy:.4f}")
        
        # Save model
        self.deep_model.save(os.path.join(self.model_path, 'deep_model_challenge.h5'))
        
        return accuracy
    
    def save_preprocessors(self):
        """Save preprocessing objects"""
        logger.info("Saving preprocessing objects...")
        
        # Save vectorizer
        joblib.dump(self.vectorizer, os.path.join(self.model_path, 'vectorizer_challenge.pkl'))
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(self.model_path, 'scaler_challenge.pkl'))
        
        # Save label encoder
        joblib.dump(self.label_encoder, os.path.join(self.model_path, 'label_encoder_challenge.pkl'))
        
        # Save tokenizer
        if self.tokenizer:
            with open(os.path.join(self.model_path, 'tokenizer_challenge.pkl'), 'wb') as f:
                pickle.dump(self.tokenizer, f)
    
    def train_models(self):
        """Main training pipeline"""
        logger.info("Starting training pipeline with Challenge-1(a) data...")
        
        # Load ground truth data
        features = self.load_ground_truth_data()
        
        if len(features) == 0:
            logger.error("No training data found!")
            return
        
        # Prepare traditional ML data
        X, y = self.prepare_training_data(features)
        
        # Train Random Forest
        rf_accuracy = self.train_random_forest(X, y)
        
        # Prepare deep learning data
        X_text, X_numerical, y_categorical = self.prepare_deep_learning_data(features)
        
        # Train deep learning model
        dl_accuracy = self.train_deep_learning_model(X_text, X_numerical, y_categorical)
        
        # Save preprocessors
        self.save_preprocessors()
        
        logger.info("Training completed!")
        logger.info(f"Random Forest Accuracy: {rf_accuracy:.4f}")
        logger.info(f"Deep Learning Accuracy: {dl_accuracy:.4f}")
        
        # Save training summary
        summary = {
            "training_samples": len(features),
            "rf_accuracy": rf_accuracy,
            "dl_accuracy": dl_accuracy,
            "label_classes": list(self.label_encoder.classes_),
            "feature_dimensions": X.shape[1]
        }
        
        with open(os.path.join(self.model_path, 'training_summary_challenge.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train ML models with Challenge-1(a) data")
    parser.add_argument("--challenge-path", 
                       default="C:/Users/mk941/Downloads/Desktop/Adobe Hackathon/Adobe-Hackathon/Challenge - 1(a)",
                       help="Path to Challenge-1(a) folder")
    parser.add_argument("--app-path",
                       default="C:/Users/mk941/Downloads/Desktop/Adobe Hackathon/Adobe-Hackathon/app",
                       help="Path to app folder")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ChallengeDataTrainer(args.challenge_path, args.app_path)
    
    # Train models
    summary = trainer.train_models()
    
    if summary:
        print("\n" + "="*50)
        print("TRAINING COMPLETE!")
        print("="*50)
        print(f"Training samples: {summary['training_samples']}")
        print(f"Random Forest accuracy: {summary['rf_accuracy']:.4f}")
        print(f"Deep Learning accuracy: {summary['dl_accuracy']:.4f}")
        print(f"Classes detected: {', '.join(summary['label_classes'])}")
        print("="*50)

if __name__ == "__main__":
    main()
