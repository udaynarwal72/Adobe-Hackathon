import fitz  # PyMuPDF
import json
import os
import re
from typing import List, Dict, Any, Optional, Tuple
import argparse
from collections import defaultdict, Counter

class EnhancedPDFOutlineExtractor:
    def __init__(self):
        # Comprehensive heading patterns
        self.heading_patterns = [
            # Primary numbered patterns
            (r'^\s*(\d+)\.\s+(.+)$', "H1"),
            (r'^\s*(\d+)\.(\d+)\s+(.+)$', "H2"),
            (r'^\s*(\d+)\.(\d+)\.(\d+)\s+(.+)$', "H3"),
            (r'^\s*(\d+)\.(\d+)\.(\d+)\.(\d+)\s+(.+)$', "H4"),
            
            # Parenthetical numbering
            (r'^\s*\((\d+)\)\s+(.+)$', "H2"),
            (r'^\s*\(([a-z])\)\s+(.+)$', "H3"),
            (r'^\s*\(([A-Z])\)\s+(.+)$', "H2"),
            
            # Chapter/Section patterns
            (r'^(?:Chapter|CHAPTER)\s+(\d+)[:\.\s]+(.+)$', "H1"),
            (r'^(?:Section|SECTION)\s+(\d+)[:\.\s]+(.+)$', "H1"),
            (r'^(?:Part|PART)\s+(\d+)[:\.\s]+(.+)$', "H1"),
            
            # Letter patterns
            (r'^\s*([A-Z])\.\s+(.+)$', "H1"),
            (r'^\s*([a-z])\.\s+(.+)$', "H2"),
            
            # Roman numerals
            (r'^\s*([IVX]{1,5})\.\s+(.+)$', "H1"),
            (r'^\s*([ivx]{1,5})\.\s+(.+)$', "H2"),
            
            # Bullet points
            (r'^\s*[•·▪▫◦‣⁃]\s+(.+)$', "H3"),
            (r'^\s*[-–—]\s+(.+)$', "H3"),
            (r'^\s*[►▶]\s+(.+)$', "H3"),
            
            # Question patterns
            (r'^\s*Q\.?\s*(\d+)[:\.]?\s+(.+)$', "H2"),
            (r'^\s*Question\s*(\d*)[:\.]?\s+(.+)$', "H2"),
            
            # Form field patterns
            (r'^\s*(.+?):\s*_+\s*$', "H3"),
            (r'^\s*(.+?):\s*Rs\.?\s*_*\s*$', "H3"),
            (r'^\s*(.+?):\s*₹\s*_*\s*$', "H3"),
            (r'^\s*Date\s*(.*):\s*_*\s*$', "H3"),
            (r'^\s*Signature\s*(.*):\s*_*\s*$', "H3"),
        ]
        
        # Heading indicators
        self.heading_indicators = {
            'introduction', 'conclusion', 'summary', 'overview', 'background',
            'application', 'form', 'details', 'information', 'declaration',
            'advance', 'amount', 'required', 'employee', 'department',
            'name', 'address', 'contact', 'phone', 'email', 'date', 'signature',
            'approval', 'sanctioned', 'granted', 'requirements', 'guidelines',
            'instructions', 'eligibility', 'criteria', 'conditions', 'terms'
        }
        
        # Body text indicators
        self.body_text_indicators = {
            'the', 'and', 'or', 'but', 'if', 'when', 'where', 'how', 'why',
            'this', 'that', 'these', 'those', 'with', 'from', 'into', 'during',
            'please', 'kindly', 'hereby', 'therefore', 'however', 'moreover',
            'should', 'would', 'could', 'might', 'will', 'shall', 'must',
            'have', 'has', 'had', 'been', 'being', 'was', 'were', 'are', 'is'
        }

    def extract_title(self, doc) -> str:
        """Extract document title"""
        try:
            # Try metadata first
            metadata = doc.metadata
            if metadata and metadata.get('title'):
                title = metadata['title'].strip()
                if self._is_valid_title(title):
                    return self._clean_title(title)
            
            # Analyze first page for title
            if len(doc) > 0:
                page = doc[0]
                text_dict = page.get_text("dict")
                
                title_candidates = []
                
                for block in text_dict.get("blocks", [])[:10]:  # Check first 10 blocks
                    if "lines" in block:
                        for line in block["lines"]:
                            line_text, max_size, formatting = self._extract_line_info(line)
                            
                            if line_text and self._could_be_title(line_text, max_size, formatting):
                                score = self._calculate_title_score(line_text, max_size, formatting)
                                title_candidates.append((score, line_text))
                
                if title_candidates:
                    title_candidates.sort(reverse=True, key=lambda x: x[0])
                    for score, title_text in title_candidates[:5]:
                        cleaned = self._clean_title(title_text)
                        if self._is_valid_title(cleaned):
                            return cleaned
            
            return "Document"
            
        except Exception as e:
            print(f"Error extracting title: {e}")
            return "Document"

    def _extract_line_info(self, line) -> Tuple[str, float, Dict]:
        """Extract information from a text line"""
        try:
            line_text = ""
            max_size = 0
            flags = 0
            
            for span in line.get("spans", []):
                line_text += span.get("text", "")
                max_size = max(max_size, span.get("size", 12))
                flags |= span.get("flags", 0)
            
            formatting = {
                'max_size': max_size,
                'is_bold': bool(flags & 2**4),
                'is_italic': bool(flags & 2**1),
                'is_underline': bool(flags & 2**0)
            }
            
            return line_text.strip(), max_size, formatting
            
        except Exception as e:
            print(f"Error extracting line info: {e}")
            return "", 12, {'max_size': 12, 'is_bold': False, 'is_italic': False, 'is_underline': False}

    def _could_be_title(self, text: str, size: float, formatting: Dict) -> bool:
        """Check if text could be a title"""
        if not text or len(text) < 2 or len(text) > 200:
            return False
        
        # Skip obvious non-titles
        skip_patterns = [
            r'^\d+$',
            r'^page\s+\d+',
            r'^\d{1,2}/\d{1,2}/\d{2,4}',
            r'^[^\w]*$'
        ]
        
        for pattern in skip_patterns:
            if re.match(pattern, text.lower()):
                return False
        
        return True

    def _calculate_title_score(self, text: str, size: float, formatting: Dict) -> float:
        """Calculate title score"""
        score = 0
        
        # Size factor
        score += size * 3
        
        # Formatting factors
        if formatting.get('is_bold', False):
            score += 30
        if formatting.get('is_underline', False):
            score += 20
        if formatting.get('is_italic', False):
            score += 10
        
        # Length factors
        word_count = len(text.split())
        if 2 <= word_count <= 10:
            score += 25
        elif word_count <= 20:
            score += 15
        
        # Content factors
        if text.istitle() or text.isupper():
            score += 20
        
        # Title-like words
        text_lower = text.lower()
        title_words = {'application', 'form', 'report', 'document', 'manual'}
        if any(word in text_lower for word in title_words):
            score += 25
        
        return score

    def _clean_title(self, title: str) -> str:
        """Clean title text"""
        # Remove common prefixes/suffixes
        title = re.sub(r'^Microsoft Word - ', '', title)
        title = re.sub(r'\.docx?$', '', title, flags=re.IGNORECASE)
        title = re.sub(r'\.pdf$', '', title, flags=re.IGNORECASE)
        
        # Clean up spacing
        title = re.sub(r'\s+', ' ', title).strip()
        
        return title

    def _is_valid_title(self, title: str) -> bool:
        """Validate title"""
        if not title or len(title) < 2 or len(title) > 200:
            return False
        
        invalid_titles = {
            'document', 'untitled', 'page', 'contents', 'index'
        }
        
        return title.lower() not in invalid_titles

    def analyze_document_structure(self, doc) -> Dict[str, Any]:
        """Analyze document structure"""
        try:
            size_analysis = defaultdict(int)
            formatting_analysis = defaultdict(int)
            
            for page_num in range(min(len(doc), 5)):  # Analyze first 5 pages
                page = doc[page_num]
                text_dict = page.get_text("dict")
                
                for block in text_dict.get("blocks", []):
                    if "lines" in block:
                        for line in block["lines"]:
                            line_text, max_size, formatting = self._extract_line_info(line)
                            
                            if line_text and len(line_text.strip()) > 0:
                                size_analysis[round(max_size, 1)] += len(line_text)
                                formatting_key = (formatting['is_bold'], formatting['is_italic'])
                                formatting_analysis[formatting_key] += 1
            
            # Determine body text size
            if size_analysis:
                body_size = max(size_analysis.items(), key=lambda x: x[1])[0]
                sizes = list(size_analysis.keys())
                sizes.sort(reverse=True)
                heading_sizes = [s for s in sizes if s > body_size * 1.1]
            else:
                body_size = 12
                heading_sizes = [14, 16, 18]
            
            return {
                'body_text_size': body_size,
                'heading_sizes': heading_sizes,
                'size_distribution': dict(size_analysis),
                'formatting_distribution': dict(formatting_analysis)
            }
            
        except Exception as e:
            print(f"Error analyzing document structure: {e}")
            return {
                'body_text_size': 12,
                'heading_sizes': [14, 16, 18],
                'size_distribution': {},
                'formatting_distribution': {}
            }

    def is_heading_advanced(self, text: str, formatting: Dict, structure_info: Dict) -> Optional[str]:
        """Advanced heading detection"""
        try:
            text = text.strip()
            
            if not text or len(text) < 1:
                return None
            
            # Skip very long text
            if len(text) > 500:
                return None
            
            # Pattern-based detection first
            for pattern, level in self.heading_patterns:
                if re.match(pattern, text, re.IGNORECASE):
                    return level
            
            # Skip obvious body text
            if self._is_likely_body_text(text):
                return None
            
            # Advanced analysis
            body_size = structure_info.get('body_text_size', 12)
            size = formatting.get('max_size', 12)
            size_ratio = size / body_size if body_size > 0 else 1
            
            is_bold = formatting.get('is_bold', False)
            is_italic = formatting.get('is_italic', False)
            is_underline = formatting.get('is_underline', False)
            
            word_count = len(text.split())
            
            # Calculate heading score
            heading_score = 0
            
            # Size factor
            if size_ratio >= 2.0:
                heading_score += 40
            elif size_ratio >= 1.5:
                heading_score += 30
            elif size_ratio >= 1.3:
                heading_score += 25
            elif size_ratio >= 1.2:
                heading_score += 20
            elif size_ratio >= 1.1:
                heading_score += 15
            elif size_ratio >= 1.05:
                heading_score += 10
            
            # Formatting factors
            if is_bold:
                heading_score += 25
            if is_underline:
                heading_score += 15
            if is_italic:
                heading_score += 5
            
            # Length factors
            if word_count == 1:
                heading_score += 20
            elif word_count <= 3:
                heading_score += 25
            elif word_count <= 8:
                heading_score += 20
            elif word_count <= 15:
                heading_score += 15
            elif word_count <= 25:
                heading_score += 10
            elif word_count > 30:
                heading_score -= 20
            
            # Content analysis
            text_lower = text.lower()
            
            # Heading indicators
            if any(indicator in text_lower for indicator in self.heading_indicators):
                heading_score += 20
            
            # Case patterns
            if text.isupper() and len(text) > 1:
                heading_score += 20
            elif text.istitle():
                heading_score += 15
            
            # Punctuation patterns
            if text.endswith(':'):
                heading_score += 15
            elif text.endswith('.') and word_count <= 5:
                heading_score += 10
            
            # Form field patterns
            if ':' in text and ('_' in text or text.endswith(':')):
                heading_score += 20
            
            # Determine level based on score and size
            if heading_score >= 50:
                if size_ratio >= 2.0:
                    return "H1"
                elif size_ratio >= 1.5:
                    return "H1"
                else:
                    return "H2"
            elif heading_score >= 35:
                if size_ratio >= 1.5:
                    return "H1"
                elif size_ratio >= 1.2:
                    return "H2"
                else:
                    return "H2"
            elif heading_score >= 25:
                if size_ratio >= 1.3:
                    return "H2"
                else:
                    return "H3"
            elif heading_score >= 15:
                return "H3"
            elif heading_score >= 10:
                return "H4"
            
            return None
            
        except Exception as e:
            print(f"Error in heading detection: {e}")
            return None

    def _is_likely_body_text(self, text: str) -> bool:
        """Check if text is likely body text"""
        try:
            text_lower = text.lower()
            
            # Common body text patterns
            body_patterns = [
                r'\b(?:this|that|the|and|or|but|if|when|where|how|why|what)\b.*\b(?:is|are|was|were|will|would|should|could)\b',
                r'\b(?:please|kindly|hereby|therefore|however|moreover)\b',
                r'[.!?]\s+[A-Z]',  # Multiple sentences
            ]
            
            for pattern in body_patterns:
                if re.search(pattern, text_lower):
                    return True
            
            # Check for high concentration of body text indicators
            words = text_lower.split()
            if len(words) > 5:
                body_word_count = sum(1 for word in words if word in self.body_text_indicators)
                if body_word_count / len(words) > 0.3:
                    return True
            
            # Skip very long sentences
            if len(words) > 25 and '.' in text:
                return True
            
            return False
            
        except Exception as e:
            print(f"Error checking body text: {e}")
            return False

    def extract_all_headings(self, doc) -> List[Dict[str, Any]]:
        """Extract all headings"""
        try:
            structure_info = self.analyze_document_structure(doc)
            headings = []
            processed_text = set()
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text_dict = page.get_text("dict")
                
                for block in text_dict.get("blocks", []):
                    if "lines" in block:
                        for line in block["lines"]:
                            line_text, max_size, formatting = self._extract_line_info(line)
                            
                            if line_text and len(line_text) >= 1:
                                # Avoid duplicates
                                text_id = (line_text.lower().strip(), page_num)
                                if text_id in processed_text:
                                    continue
                                processed_text.add(text_id)
                                
                                heading_level = self.is_heading_advanced(line_text, formatting, structure_info)
                                
                                if heading_level:
                                    clean_text = self._clean_heading_text(line_text)
                                    
                                    if self._is_valid_heading(clean_text):
                                        headings.append({
                                            "level": heading_level,
                                            "text": clean_text,
                                            "page": page_num + 1
                                        })
            
            return self._refine_headings(headings)
            
        except Exception as e:
            print(f"Error extracting headings: {e}")
            return []

    def _clean_heading_text(self, text: str) -> str:
        """Clean heading text"""
        try:
            original_text = text.strip()
            
            # Remove numbered patterns
            patterns_to_remove = [
                r'^\s*\d+(\.\d+)*\.?\s+',
                r'^\s*\(\d+\)\s+',
                r'^\s*\([a-zA-Z]\)\s+',
                r'^\s*[A-Z]\.?\s+(?=[A-Z])',
                r'^\s*[a-z]\.?\s+(?=[a-z])',
                r'^\s*[IVX]{1,5}\.?\s+',
                r'^\s*[ivx]{1,5}\.?\s+',
                r'^\s*[•·▪▫◦‣⁃►▶-–—]\s+',
                r'^\s*Q\.?\s*\d*[:\.]?\s+',
                r'^\s*Question\s*\d*[:\.]?\s+',
            ]
            
            cleaned = original_text
            for pattern in patterns_to_remove:
                cleaned = re.sub(pattern, '', cleaned)
            
            # Clean up spacing
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            
            # If cleaning removed too much, return original
            if len(cleaned) < len(original_text) * 0.2:
                return original_text
            
            return cleaned if cleaned else original_text
            
        except Exception as e:
            print(f"Error cleaning heading text: {e}")
            return text

    def _is_valid_heading(self, text: str) -> bool:
        """Validate heading"""
        if not text or len(text) < 1:
            return False
        
        # Skip pure numbers or symbols
        if re.match(r'^[\d\s\.\-_]+$', text) and len(text) > 3:
            return False
        
        return True

    def _refine_headings(self, headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Refine headings"""
        try:
            if not headings:
                return headings
            
            # Sort by page and text
            headings.sort(key=lambda x: (x['page'], x['text']))
            
            # Remove exact duplicates
            refined = []
            seen_texts = set()
            
            for heading in headings:
                text_key = heading['text'].lower().strip()
                
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    refined.append(heading)
            
            return refined
            
        except Exception as e:
            print(f"Error refining headings: {e}")
            return headings

    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process PDF"""
        try:
            print(f"Opening PDF: {pdf_path}")
            doc = fitz.open(pdf_path)
            
            # Extract title
            print("Extracting title...")
            title = self.extract_title(doc)
            
            # Extract headings
            print("Extracting headings...")
            headings = self.extract_all_headings(doc)
            
            doc.close()
            
            print(f"Extracted {len(headings)} headings")
            
            return {
                "title": title,
                "outline": headings
            }
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return {
                "title": "Error",
                "outline": []
            }

def main():
    parser = argparse.ArgumentParser(description='Enhanced PDF outline extraction')
    parser.add_argument('--input', required=True, help='Input directory containing PDF files')
    parser.add_argument('--output', required=True, help='Output directory for JSON files')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    # Convert to absolute paths
    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Check input directory
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist!")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    extractor = EnhancedPDFOutlineExtractor()
    
    # Find PDF files
    pdf_files = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.pdf'):
            pdf_files.append(filename)
    
    if not pdf_files:
        print(f"No PDF files found in '{input_dir}'")
        return
    
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

if __name__ == "__main__":
    main()