#!/usr/bin/env python3
"""
Evaluation script to compare ML model predictions with expected results
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

def load_json(file_path: str) -> Dict:
    """Load JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}

def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    return text.lower().strip()

def calculate_heading_accuracy(predicted: List[Dict], expected: List[Dict]) -> Dict:
    """Calculate accuracy metrics for headings"""
    if not expected:
        return {
            'total_expected': 0,
            'total_predicted': 0,
            'correct_headings': 0,
            'correct_levels': 0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'level_accuracy': 0.0
        }
    
    # Create mappings for comparison
    expected_map = {(normalize_text(h['text']), h['page']): h['level'] for h in expected}
    predicted_map = {(normalize_text(h['text']), h['page']): h['level'] for h in predicted}
    
    # Calculate metrics
    correct_headings = 0
    correct_levels = 0
    total_expected = len(expected)
    total_predicted = len(predicted)
    
    # Check each expected heading
    for key, expected_level in expected_map.items():
        if key in predicted_map:
            correct_headings += 1
            if predicted_map[key] == expected_level:
                correct_levels += 1
    
    # Calculate precision, recall, F1
    precision = correct_headings / total_predicted if total_predicted > 0 else 0
    recall = correct_headings / total_expected if total_expected > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    level_accuracy = correct_levels / total_expected if total_expected > 0 else 0
    
    return {
        'total_expected': total_expected,
        'total_predicted': total_predicted,
        'correct_headings': correct_headings,
        'correct_levels': correct_levels,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'level_accuracy': level_accuracy
    }

def evaluate_single_file(predicted_file: str, expected_file: str) -> Dict:
    """Evaluate a single file"""
    predicted = load_json(predicted_file)
    expected = load_json(expected_file)
    
    if not predicted or not expected:
        return {}
    
    # Title accuracy
    predicted_title = normalize_text(predicted.get('title', ''))
    expected_title = normalize_text(expected.get('title', ''))
    title_match = predicted_title == expected_title
    
    # Heading accuracy
    predicted_outline = predicted.get('outline', [])
    expected_outline = expected.get('outline', [])
    
    heading_metrics = calculate_heading_accuracy(predicted_outline, expected_outline)
    
    return {
        'title_match': title_match,
        'predicted_title': predicted.get('title', ''),
        'expected_title': expected.get('title', ''),
        **heading_metrics
    }

def evaluate_batch(predicted_dir: str, expected_dir: str) -> Dict:
    """Evaluate multiple files"""
    predicted_path = Path(predicted_dir)
    expected_path = Path(expected_dir)
    
    results = {}
    overall_metrics = {
        'total_files': 0,
        'title_matches': 0,
        'total_expected_headings': 0,
        'total_predicted_headings': 0,
        'total_correct_headings': 0,
        'total_correct_levels': 0,
        'title_accuracy': 0.0,
        'overall_recall': 0.0,
        'overall_level_accuracy': 0.0,
        'overall_precision': 0.0,
        'overall_f1': 0.0
    }
    
    # Process each file
    for predicted_file in predicted_path.glob('*.json'):
        expected_file = expected_path / predicted_file.name
        
        if expected_file.exists():
            file_result = evaluate_single_file(str(predicted_file), str(expected_file))
            
            if file_result:
                results[predicted_file.stem] = file_result
                
                # Update overall metrics
                overall_metrics['total_files'] += 1
                if file_result['title_match']:
                    overall_metrics['title_matches'] += 1
                
                overall_metrics['total_expected_headings'] += file_result['total_expected']
                overall_metrics['total_predicted_headings'] += file_result['total_predicted']
                overall_metrics['total_correct_headings'] += file_result['correct_headings']
                overall_metrics['total_correct_levels'] += file_result['correct_levels']
    
    # Calculate overall metrics
    if overall_metrics['total_files'] > 0:
        overall_metrics['title_accuracy'] = overall_metrics['title_matches'] / overall_metrics['total_files']
    
    if overall_metrics['total_expected_headings'] > 0:
        overall_metrics['overall_recall'] = overall_metrics['total_correct_headings'] / overall_metrics['total_expected_headings']
        overall_metrics['overall_level_accuracy'] = overall_metrics['total_correct_levels'] / overall_metrics['total_expected_headings']
    
    if overall_metrics['total_predicted_headings'] > 0:
        overall_metrics['overall_precision'] = overall_metrics['total_correct_headings'] / overall_metrics['total_predicted_headings']
    
    if overall_metrics['overall_precision'] > 0 and overall_metrics['overall_recall'] > 0:
        p = overall_metrics['overall_precision']
        r = overall_metrics['overall_recall']
        overall_metrics['overall_f1'] = 2 * (p * r) / (p + r)
    
    return {
        'individual_results': results,
        'overall_metrics': overall_metrics
    }

def print_results(results: Dict):
    """Print evaluation results"""
    overall = results['overall_metrics']
    individual = results['individual_results']
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nOverall Metrics:")
    print(f"  Files evaluated: {overall['total_files']}")
    print(f"  Title accuracy: {overall.get('title_accuracy', 0):.3f}")
    print(f"  Heading precision: {overall.get('overall_precision', 0):.3f}")
    print(f"  Heading recall: {overall.get('overall_recall', 0):.3f}")
    print(f"  Heading F1-score: {overall.get('overall_f1', 0):.3f}")
    print(f"  Level accuracy: {overall.get('overall_level_accuracy', 0):.3f}")
    
    print(f"\nDetailed Statistics:")
    print(f"  Expected headings: {overall['total_expected_headings']}")
    print(f"  Predicted headings: {overall['total_predicted_headings']}")
    print(f"  Correctly identified: {overall['total_correct_headings']}")
    print(f"  Correct levels: {overall['total_correct_levels']}")
    
    print(f"\nPer-file Results:")
    for filename, result in individual.items():
        print(f"  {filename}:")
        print(f"    Title match: {'✓' if result['title_match'] else '✗'}")
        print(f"    Heading F1: {result['f1_score']:.3f}")
        print(f"    Level accuracy: {result['level_accuracy']:.3f}")
        print(f"    Precision: {result['precision']:.3f}")
        print(f"    Recall: {result['recall']:.3f}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate ML PDF Extractor results')
    parser.add_argument('--predicted', required=True, help='Directory containing predicted JSON files')
    parser.add_argument('--expected', required=True, help='Directory containing expected JSON files')
    parser.add_argument('--output', help='Output file for detailed results (JSON)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.predicted):
        print(f"Error: Predicted directory '{args.predicted}' does not exist")
        return 1
    
    if not os.path.exists(args.expected):
        print(f"Error: Expected directory '{args.expected}' does not exist")
        return 1
    
    # Run evaluation
    results = evaluate_batch(args.predicted, args.expected)
    
    # Print results
    print_results(results)
    
    # Save detailed results if requested
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {args.output}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
