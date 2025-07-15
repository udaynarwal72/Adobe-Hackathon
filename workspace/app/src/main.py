import fitz  # PyMuPDF
import json
import os
import re
from typing import List, Dict, Any
import argparse

class PDFOutlineExtractor:
    def __init__(self):
        self.heading_patterns = [
            # Common heading patterns
            r'^\d+\.\s+(.+)$',  # 1. Title
            r'^\d+\.\d+\s+(.+)$',  # 1.1 Subtitle
            r'^\d+\.\d+\.\d+\s+(.+)$',  # 1.1.1 Sub-subtitle
            r'^([A-Z][A-Z\s]+)$',  # ALL CAPS
            r'^([A-Z][a-z\s]+)$',  # Title Case
            r'^Chapter\s+\d+[:\s]+(.+)$',  # Chapter patterns
            r'^Section\s+\d+[:\s]+(.+)$',  # Section patterns
        ]
        
    def extract_title(self, doc) -> str:
        """Extract document title from various sources"""
        # Try metadata first
        metadata = doc.metadata
        if metadata.get('title'):
            return metadata['title'].strip()
        
        # Try first page for title
        first_page = doc[0]
        text_dict = first_page.get_text("dict")
        
        # Look for largest font size text in first few blocks
        max_font_size = 0
        title_candidate = ""
        
        for block in text_dict["blocks"][:5]:  # Check first 5 blocks
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        if span["size"] > max_font_size:
                            max_font_size = span["size"]
                            title_candidate = span["text"].strip()
        
        if title_candidate and len(title_candidate) > 3:
            return title_candidate
        
        # Fallback to filename
        return "Document"
    
    def analyze_text_formatting(self, doc) -> Dict[str, Any]:
        """Analyze document formatting to identify heading patterns"""
        font_sizes = {}
        font_flags = {}
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text_dict = page.get_text("dict")
            
            for block in text_dict["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            size = span["size"]
                            flags = span["flags"]
                            text = span["text"].strip()
                            
                            if text and len(text) > 2:
                                if size not in font_sizes:
                                    font_sizes[size] = []
                                font_sizes[size].append(text)
                                
                                if flags not in font_flags:
                                    font_flags[flags] = []
                                font_flags[flags].append(text)
        
        # Sort font sizes to identify heading hierarchy
        sorted_sizes = sorted(font_sizes.keys(), reverse=True)
        
        return {
            'font_sizes': font_sizes,
            'font_flags': font_flags,
            'size_hierarchy': sorted_sizes
        }
    
    def is_heading(self, text: str, size: float, flags: int, formatting_info: Dict) -> str:
        """Determine if text is a heading and its level"""
        text = text.strip()
        
        if not text or len(text) < 3:
            return None
        
        # Skip if too long (likely paragraph)
        if len(text) > 200:
            return None
        
        # Check against regex patterns
        for pattern in self.heading_patterns:
            if re.match(pattern, text):
                # Determine level based on pattern and font size
                if re.match(r'^\d+\.\s+', text):
                    return "H1"
                elif re.match(r'^\d+\.\d+\s+', text):
                    return "H2"
                elif re.match(r'^\d+\.\d+\.\d+\s+', text):
                    return "H3"
        
        # Font size based heading detection
        size_hierarchy = formatting_info['size_hierarchy']
        if len(size_hierarchy) >= 3:
            if size >= size_hierarchy[0]:
                return "H1"
            elif size >= size_hierarchy[1]:
                return "H2"
            elif size >= size_hierarchy[2]:
                return "H3"
        
        # Bold/formatting based detection
        if flags & 2**4:  # Bold flag
            if size > 12:
                return "H1"
            elif size > 10:
                return "H2"
            else:
                return "H3"
        
        # All caps detection
        if text.isupper() and len(text.split()) <= 10:
            return "H1"
        
        return None
    
    def extract_headings(self, doc) -> List[Dict[str, Any]]:
        """Extract headings from PDF document"""
        formatting_info = self.analyze_text_formatting(doc)
        headings = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text_dict = page.get_text("dict")
            
            for block in text_dict["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        line_size = 0
                        line_flags = 0
                        
                        for span in line["spans"]:
                            line_text += span["text"]
                            line_size = max(line_size, span["size"])
                            line_flags |= span["flags"]
                        
                        line_text = line_text.strip()
                        if line_text:
                            heading_level = self.is_heading(line_text, line_size, line_flags, formatting_info)
                            if heading_level:
                                # Clean up the text
                                clean_text = re.sub(r'^\d+\.?\s*', '', line_text)
                                clean_text = re.sub(r'^Chapter\s+\d+[:\s]+', '', clean_text)
                                clean_text = re.sub(r'^Section\s+\d+[:\s]+', '', clean_text)
                                clean_text = clean_text.strip()
                                
                                if clean_text:
                                    headings.append({
                                        "level": heading_level,
                                        "text": clean_text,
                                        "page": page_num + 1
                                    })
        
        return headings
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process a single PDF file"""
        try:
            doc = fitz.open(pdf_path)
            
            # Extract title
            title = self.extract_title(doc)
            
            # Extract headings
            headings = self.extract_headings(doc)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_headings = []
            for heading in headings:
                heading_key = (heading['level'], heading['text'], heading['page'])
                if heading_key not in seen:
                    seen.add(heading_key)
                    unique_headings.append(heading)
            
            doc.close()
            
            return {
                "title": title,
                "outline": unique_headings
            }
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return {
                "title": "Error",
                "outline": []
            }

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