import fitz  # PyMuPDF
import json
import os
import re
from typing import List, Dict, Any, Optional
import argparse

class PDFOutlineExtractor:
    def __init__(self):
        self.heading_patterns = [
            # Numbered patterns (most reliable)
            r'^\s*(\d+)\.\s+(.+)$',  # 1. Title
            r'^\s*(\d+)\.(\d+)\s+(.+)$',  # 1.1 Subtitle  
            r'^\s*(\d+)\.(\d+)\.(\d+)\s+(.+)$',  # 1.1.1 Sub-subtitle
            r'^\s*(\d+)\.(\d+)\.(\d+)\.(\d+)\s+(.+)$',  # 1.1.1.1 Sub-sub-subtitle
            
            # Chapter/Section patterns
            r'^(?:Chapter|CHAPTER)\s+(\d+)[:\.\s]+(.+)$',
            r'^(?:Section|SECTION)\s+(\d+)[:\.\s]+(.+)$',
            r'^(?:Part|PART)\s+(\d+)[:\.\s]+(.+)$',
            
            # Letter patterns
            r'^\s*([A-Z])\.\s+(.+)$',  # A. Title
            r'^\s*([a-z])\.\s+(.+)$',  # a. subtitle
            
            # Roman numerals
            r'^\s*([IVX]+)\.\s+(.+)$',  # I. Title, II. Title
            r'^\s*([ivx]+)\.\s+(.+)$',  # i. subtitle
        ]
        
        # Font size thresholds for different heading levels
        self.font_size_thresholds = {
            'title': 16,
            'h1': 14,
            'h2': 12,
            'h3': 10
        }
        
    def extract_title(self, doc) -> str:
        """Extract document title with enhanced accuracy and better cleaning"""
        # Try metadata first with better validation
        metadata = doc.metadata
        if metadata and metadata.get('title'):
            title = metadata['title'].strip()
            # Clean up common metadata artifacts and validate
            if (title and 
                not title.lower().endswith('.pdf') and 
                not title.lower().startswith('microsoft word') and
                len(title) > 2 and
                len(title) < 200):
                # Clean the metadata title
                title = re.sub(r'^Microsoft Word - ', '', title)  # Remove Word prefix
                title = re.sub(r'\.docx?$', '', title)  # Remove doc extensions
                title = title.strip()
                if len(title) > 3:
                    return title
        
        # Analyze first few pages more comprehensively for title
        title_candidates = []
        
        for page_num in range(min(3, len(doc))):  # Check first 3 pages
            page = doc[page_num]
            text_dict = page.get_text("dict")
            
            for block_idx, block in enumerate(text_dict["blocks"][:10]):  # Check first 10 blocks per page
                if "lines" in block:
                    for line_idx, line in enumerate(block["lines"]):
                        line_text = ""
                        max_size = 0
                        is_bold = False
                        is_centered = False
                        
                        for span in line["spans"]:
                            line_text += span["text"]
                            max_size = max(max_size, span["size"])
                            if span["flags"] & 2**4:  # Bold flag
                                is_bold = True
                        
                        line_text = line_text.strip()
                        
                        # Enhanced title criteria
                        if (line_text and 
                            len(line_text) > 3 and 
                            len(line_text) < 200 and  # Not too long
                            max_size >= 12):  # Reasonable font size
                            
                            # Skip obvious non-title content
                            if (line_text.lower() in ['page', 'contents', 'table of contents', 'index'] or
                                re.match(r'^\d+$', line_text) or  # Just page numbers
                                re.match(r'^page\s+\d+', line_text.lower()) or
                                len(line_text.split()) > 15):  # Too many words
                                continue
                            
                            # Calculate score based on multiple factors
                            score = 0
                            
                            # Size factor
                            score += max_size * 2
                            
                            # Position factor (earlier is better)
                            score += max(0, 50 - (page_num * 20 + block_idx * 5 + line_idx))
                            
                            # Formatting factor
                            if is_bold:
                                score += 15
                            
                            # Length factor (moderate length preferred)
                            if 10 <= len(line_text) <= 80:
                                score += 10
                            elif 5 <= len(line_text) <= 150:
                                score += 5
                            
                            # Content factor
                            if (line_text.istitle() or 
                                line_text.isupper() or 
                                any(word.istitle() for word in line_text.split())):
                                score += 10
                            
                            # Avoid obviously bad titles
                            bad_indicators = ['overview', 'version', 'revision', 'history', 'acknowledgements']
                            if any(indicator in line_text.lower() for indicator in bad_indicators):
                                score -= 20
                            
                            title_candidates.append((score, line_text, max_size, page_num))
        
        # Sort by score and get the best candidate
        if title_candidates:
            title_candidates.sort(reverse=True, key=lambda x: x[0])
            
            # Get the best candidate but validate it
            for score, best_title, size, page in title_candidates[:3]:  # Check top 3
                # Clean up the title carefully
                cleaned_title = best_title.strip()
                
                # Only remove numbered prefixes if they're clearly list items
                if re.match(r'^\d+\.\s+', cleaned_title):
                    cleaned_title = re.sub(r'^\d+\.\s+', '', cleaned_title)
                
                # Remove common artifacts but preserve content
                cleaned_title = re.sub(r'^[^\w]*', '', cleaned_title)  # Leading non-word chars
                cleaned_title = re.sub(r'\s+', ' ', cleaned_title)  # Multiple spaces
                cleaned_title = cleaned_title.strip()
                
                # Final validation
                if (cleaned_title and 
                    len(cleaned_title) > 3 and
                    not re.match(r'^\d+$', cleaned_title) and
                    cleaned_title.lower() not in ['overview', 'introduction', 'contents']):
                    return cleaned_title
        
        # Enhanced fallback: try to construct from filename or first meaningful text
        first_page = doc[0]
        text_dict = first_page.get_text("dict")
        
        # Look for any reasonable text as last resort
        for block in text_dict["blocks"][:5]:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if (text and 
                            len(text) > 5 and 
                            len(text) < 100 and
                            not re.match(r'^\d+$', text)):
                            return text
        
        return "Document"
    
    def analyze_text_formatting(self, doc) -> Dict[str, Any]:
        """Analyze document formatting with improved statistical analysis"""
        font_sizes = {}
        font_flags = {}
        text_blocks = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text_dict = page.get_text("dict")
            
            for block in text_dict["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            size = round(span["size"], 1)  # Round to avoid floating point issues
                            flags = span["flags"]
                            text = span["text"].strip()
                            
                            if text and len(text) > 1:
                                if size not in font_sizes:
                                    font_sizes[size] = []
                                font_sizes[size].append(text)
                                
                                if flags not in font_flags:
                                    font_flags[flags] = []
                                font_flags[flags].append(text)
                                
                                # Store for analysis
                                text_blocks.append({
                                    'text': text,
                                    'size': size,
                                    'flags': flags,
                                    'page': page_num + 1
                                })
        
        # Enhanced font size analysis
        sorted_sizes = sorted(font_sizes.keys(), reverse=True)
        
        # Find most common font size (likely body text)
        size_counts = {size: len(texts) for size, texts in font_sizes.items()}
        body_text_size = max(size_counts.items(), key=lambda x: x[1])[0] if size_counts else 12
        
        # Calculate heading size thresholds based on document statistics
        heading_sizes = [s for s in sorted_sizes if s > body_text_size]
        
        return {
            'font_sizes': font_sizes,
            'font_flags': font_flags,
            'size_hierarchy': sorted_sizes,
            'body_text_size': body_text_size,
            'heading_sizes': heading_sizes,
            'text_blocks': text_blocks
        }
    
    def is_heading(self, text: str, size: float, flags: int, formatting_info: Dict) -> Optional[str]:
        """Determine if text is a heading with improved accuracy and level detection"""
        text = text.strip()
        
        if not text or len(text) < 2:
            return None
        
        # Skip very long text (likely paragraphs)
        if len(text) > 300:
            return None
            
        # Skip single characters or very short text unless it's clearly numbered
        if len(text) <= 2 and not re.match(r'^\d+\.?\s*[A-Za-z]', text):
            return None
        
        # Get document statistics
        body_text_size = formatting_info.get('body_text_size', 12)
        heading_sizes = formatting_info.get('heading_sizes', [])
        
        # Enhanced numbered pattern detection (most reliable)
        numbered_patterns = [
            (r'^\s*\d+\.\s+\w+', "H1"),  # 1. Introduction
            (r'^\s*\d+\.\d+\s+\w+', "H2"),  # 1.1 Overview
            (r'^\s*\d+\.\d+\.\d+\s+\w+', "H3"),  # 1.1.1 Details
        ]
        
        for pattern, level in numbered_patterns:
            if re.match(pattern, text):
                return level
        
        # Chapter/Section patterns
        if re.match(r'^\s*(?:Chapter|CHAPTER)\s+\d+', text):
            return "H1"
        elif re.match(r'^\s*(?:Section|SECTION)\s+\d+', text):
            return "H2"
        elif re.match(r'^\s*(?:Part|PART)\s+\d+', text):
            return "H1"
        
        # Letter patterns (be more specific)
        if re.match(r'^\s*[A-Z]\.\s+[A-Z]', text):  # A. Introduction
            return "H1"
        elif re.match(r'^\s*[a-z]\.\s+[a-z]', text):  # a. overview
            return "H2"
        
        # Roman numeral patterns
        if re.match(r'^\s*[IVX]{1,4}\.\s+\w+', text):
            return "H1"
        elif re.match(r'^\s*[ivx]{1,4}\.\s+\w+', text):
            return "H2"
        
        # Font size and formatting based detection (improved thresholds)
        is_bold = bool(flags & 2**4)
        is_italic = bool(flags & 2**1)
        
        # Calculate relative size more accurately
        size_ratio = size / body_text_size if body_text_size > 0 else 1
        
        # Skip if text looks like body text (common phrases that shouldn't be headings)
        body_text_indicators = [
            r'\b(?:this|that|the|and|or|but|if|when|where|how|why|what)\b',
            r'\b(?:page|pages|figure|table|see|refer|according|however)\b',
            r'[.!?]\s+[A-Z]',  # Contains sentence-ending punctuation followed by capital
            r'\b(?:in|on|at|for|with|by|from|to|of|as)\s+\w+\s+\w+',  # Common prepositions with context
        ]
        
        text_lower = text.lower()
        for indicator in body_text_indicators:
            if re.search(indicator, text_lower):
                # Be more strict for potential body text
                if not (is_bold and size_ratio >= 1.3):
                    return None
        
        # Enhanced heading detection with better thresholds
        if size_ratio >= 1.6 or size >= 16:  # Very large text
            if is_bold or size_ratio >= 2.0:
                return "H1"
            elif len(text.split()) <= 8:  # Short enough to be a heading
                return "H1"
        elif size_ratio >= 1.3 or size >= 14:  # Large text
            if is_bold:
                return "H1"
            elif len(text.split()) <= 6:
                return "H2"
        elif size_ratio >= 1.15 or size >= 12:  # Medium-large text
            if is_bold and len(text.split()) <= 8:
                return "H2"
            elif is_bold and len(text.split()) <= 5:
                return "H1"
        elif size_ratio >= 1.05 or size >= 11:  # Slightly larger text
            if is_bold and len(text.split()) <= 6:
                return "H3"
        
        # All caps detection (more restrictive)
        if (text.isupper() and 
            3 <= len(text) <= 80 and 
            len(text.split()) <= 8 and
            size_ratio >= 1.1):
            if size_ratio >= 1.4:
                return "H1"
            elif size_ratio >= 1.2:
                return "H2"
            else:
                return "H3"
        
        # Title case detection (be more selective)
        if (text.istitle() and 
            len(text.split()) <= 8 and 
            len(text) <= 100 and
            not text.endswith('.') and 
            is_bold and
            size_ratio >= 1.2):
            if size_ratio >= 1.4:
                return "H1"
            elif size_ratio >= 1.25:
                return "H2"
            else:
                return "H3"
        
        return None
    
    def extract_headings(self, doc) -> List[Dict[str, Any]]:
        """Extract headings with improved accuracy and proper text preservation"""
        formatting_info = self.analyze_text_formatting(doc)
        headings = []
        seen_text_per_page = {}  # Track text per page to avoid duplicates
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text_dict = page.get_text("dict")
            
            if page_num not in seen_text_per_page:
                seen_text_per_page[page_num] = set()
            
            for block in text_dict["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        line_size = 0
                        line_flags = 0
                        
                        # Combine spans in the line
                        for span in line["spans"]:
                            line_text += span["text"]
                            line_size = max(line_size, span["size"])
                            line_flags |= span["flags"]
                        
                        line_text = line_text.strip()
                        
                        if line_text and len(line_text) >= 2:
                            # Avoid processing the same text multiple times on the same page
                            text_key = line_text.lower()
                            if text_key in seen_text_per_page[page_num]:
                                continue
                            seen_text_per_page[page_num].add(text_key)
                            
                            heading_level = self.is_heading(line_text, line_size, line_flags, formatting_info)
                            if heading_level:
                                # IMPROVED: More careful text cleaning that preserves actual content
                                clean_text = line_text.strip()
                                
                                # Only remove SPECIFIC numbered patterns, not any leading characters
                                # Remove numbered list patterns like "1. ", "1.1 ", "1.1.1 "
                                clean_text = re.sub(r'^\s*\d+(\.\d+)*\.?\s+', '', clean_text)
                                
                                # Remove chapter/section patterns more carefully
                                clean_text = re.sub(r'^(?:Chapter|CHAPTER)\s+\d+[:\.\s]+', '', clean_text)
                                clean_text = re.sub(r'^(?:Section|SECTION)\s+\d+[:\.\s]+', '', clean_text)
                                clean_text = re.sub(r'^(?:Part|PART)\s+\d+[:\.\s]+', '', clean_text)
                                
                                # Remove letter patterns like "A. ", "a. " but be more specific
                                clean_text = re.sub(r'^\s*[A-Z]\.?\s+(?=[A-Z])', '', clean_text)  # Only if followed by uppercase
                                clean_text = re.sub(r'^\s*[a-z]\.?\s+(?=[a-z])', '', clean_text)  # Only if followed by lowercase
                                
                                # Remove roman numeral patterns more carefully
                                clean_text = re.sub(r'^\s*[IVX]{1,4}\.?\s+', '', clean_text)
                                clean_text = re.sub(r'^\s*[ivx]{1,4}\.?\s+', '', clean_text)
                                
                                # Clean up spacing but preserve the actual content
                                clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                                
                                # Only add if text is meaningful and not just artifacts
                                if (clean_text and 
                                    len(clean_text) >= 2 and 
                                    not re.match(r'^\d+$', clean_text) and  # Not just numbers
                                    clean_text.lower() not in ['page', 'figure', 'table', 'contents', 'index']):
                                    
                                    headings.append({
                                        "level": heading_level,
                                        "text": clean_text,
                                        "page": page_num + 1
                                    })
        
        # Enhanced deduplication while preserving order and page info
        unique_headings = []
        seen_combinations = set()
        
        for heading in headings:
            # Create a key that considers level and normalized text
            normalized_text = re.sub(r'\s+', ' ', heading['text'].lower().strip())
            heading_key = (heading['level'], normalized_text)
            
            # Also check for very similar headings (avoid near-duplicates)
            is_duplicate = False
            for existing_key in seen_combinations:
                if (existing_key[0] == heading['level'] and 
                    self._texts_are_similar(existing_key[1], normalized_text)):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_combinations.add(heading_key)
                unique_headings.append(heading)
        
        return unique_headings
    
    def _texts_are_similar(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """Check if two texts are similar enough to be considered duplicates"""
        if text1 == text2:
            return True
        
        # Simple similarity check based on common words
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return False
        
        common_words = words1.intersection(words2)
        total_words = words1.union(words2)
        
        similarity = len(common_words) / len(total_words)
        return similarity >= threshold
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process a single PDF file with enhanced accuracy"""
        try:
            doc = fitz.open(pdf_path)
            
            # Extract title with improved method
            title = self.extract_title(doc)
            
            # Extract headings with enhanced detection
            headings = self.extract_headings(doc)
            
            # Sort headings by page number, then by appearance order
            headings.sort(key=lambda x: x['page'])
            
            # Post-process to ensure logical hierarchy
            processed_headings = self._post_process_headings(headings)
            
            doc.close()
            
            return {
                "title": title,
                "outline": processed_headings
            }
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return {
                "title": "Error",
                "outline": []
            }
    
    def _post_process_headings(self, headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Post-process headings to ensure logical hierarchy and remove noise"""
        if not headings:
            return headings
        
        processed = []
        
        for heading in headings:
            # Skip very short headings that might be noise
            if len(heading['text']) < 2:
                continue
                
            # Skip headings that look like page numbers or references
            if re.match(r'^\d+$', heading['text'].strip()):
                continue
                
            # Skip common PDF artifacts
            if heading['text'].lower() in ['page', 'figure', 'table', 'contents', 'index']:
                continue
            
            processed.append(heading)
        
        return processed

def main():
    parser = argparse.ArgumentParser(description='Extract PDF outline')
    parser.add_argument('--input', default='/app/input', help='Input directory')
    parser.add_argument('--output', default='/app/output', help='Output directory')
    args = parser.parse_args()
    
    extractor = PDFOutlineExtractor()
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    # Process all PDF files in input directory
    for filename in os.listdir(args.input):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(args.input, filename)
            output_path = os.path.join(args.output, filename.replace('.pdf', '.json'))
            
            print(f"Processing {filename}...")
            result = extractor.process_pdf(pdf_path)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"Saved outline to {output_path}")

if __name__ == "__main__":
    main()