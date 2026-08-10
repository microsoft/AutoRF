import os
import re
import io
from collections import defaultdict
import json



class PDFExtractor:
    def __init__(self, pdf_path):
        """Initialize the PDF extractor with a PDF file path."""
        import pymupdf as fitz  # PyMuPDF (optional dependency, imported lazily)
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        
    def extract_content(self):
        """Extract the structured content of the PDF."""
        content = {
            'metadata': self._extract_metadata(),
            'main_text': self._extract_main_text(),
            # 'figures': self._extract_figures(), # Commented out for now, no need for figures
            'references': self._extract_references()
        }
        
        self.content = content
        return content
        
    def _extract_metadata(self):
        """Extract metadata like title, authors, abstract."""
        metadata = {
            'title': '',
            # 'authors': [],
            'abstract': '',
            # 'keywords': []
        }
        
        # Get first page text
        first_page = self.doc[0].get_text()
        
        # Extract title (usually the first line of text)
        lines = first_page.split('\n')
        metadata['title'] = lines[0].strip()
        
        # Try to find abstract or ABSTRACT or Abstract
        # Abstract with line breaks or decorations
        abstract_match = re.search( r'(?i:ABSTRACT|Abstract)(.*?)(?=\n\n|\n[A-Z][a-z])',
                            first_page, re.DOTALL)
        if abstract_match:
            metadata['abstract'] = abstract_match.group(1).strip()
            
        # # Try to find keywords
        # keywords_match = re.search(r'Keywords?[:\.]\s*(.*?)(?=\n\n)', 
        #                          first_page, re.DOTALL)
        # if keywords_match:
        #     keywords = keywords_match.group(1).strip()
        #     metadata['keywords'] = [k.strip() for k in keywords.split(',')]
            
        return metadata
    
    def _extract_main_text(self):
        """Extract main sections and their content."""
        main_text = ""
        current_section = 'Introduction'

        start_flag = False
        
        for page in self.doc:
            text = page.get_text()
            
            # if find "Introduction" or "INTRODUCTION" case sensitive
            intro_match = re.search(r'(Introduction|INTRODUCTION)', text)
            if intro_match:
                start_flag = True
                main_text += text[intro_match.end():].strip()
                continue

            # if find "References" or "REFERENCES", case sensitive
            ref_match = re.search(r'(References|REFERENCES)', text)
            if ref_match:
                start_flag = False
                # add the text before "References" to the main_text
                main_text += text[:ref_match.start()].strip()
                break
            
            if start_flag:
                main_text += text

                
        return main_text

    # def _is_section_header(self, line):
    #     """
    #     Determine if a line is likely a section header using multiple heuristics.
    #     Returns: bool
    #     """
    #     # Remove leading/trailing whitespace
    #     line = line.strip()
        
    #     # Skip if line is too long (section headers are usually brief)
    #     if len(line) > 100:
    #         return False
        
    #     # Common section header patterns
    #     patterns = [
    #         # Standard numbered sections (1. Introduction, 2.1 Methods)
    #         r'^[0-9\.]+\s+[A-Z][a-zA-Z\s]{2,}$',
            
    #         # Roman numeral sections (I. Introduction, IV. Results)
    #         r'^[IVX]+\.\s+[A-Z][a-zA-Z\s]{2,}$',
            
    #         # # Simple capitalized headers (Introduction, Materials and Methods)
    #         # r'^[A-Z][a-zA-Z\s]{2,}$',
            
    #         # Letter-numbered sections (A. Methods, B.1 Analysis)
    #         r'^[A-Z]\.?[0-9]*\s+[A-Z][a-zA-Z\s]{2,}$',
            
    #         # Appendix sections (Appendix A:, Appendix B.1)
    #         r'^Appendix\s+[A-Z](?:\.[0-9]+)?(?:\:|\.)?\s*[A-Z][a-zA-Z\s]*$'
    #     ]
        
    #     # Check against each pattern
    #     for pattern in patterns:
    #         if re.match(pattern, line):
    #             return True
                
    #     # Additional heuristics
        
    #     # Check if surrounded by blank lines (context-based)
    #     # Note: This would require passing surrounding lines as parameters
        
    #     # Check if followed by subsections
    #     # Note: This would require passing next few lines as parameters
        
    #     # # Check common section words
    #     # common_sections = {
    #     #     'introduction', 'background', 'methods', 'methodology', 'results',
    #     #     'discussion', 'conclusion', 'references', 'acknowledgments',
    #     #     'abstract', 'materials', 'experimental', 'analysis'
    #     # }
        
    #     # # If line contains any common section word and starts with capital
    #     # words = line.lower().split()
    #     # if (any(word in common_sections for word in words) and 
    #     #     line[0].isupper() and 
    #     #     len(words) <= 5):  # Section headers are usually short
    #     #     return True
            
    #     return False
      
    # def _extract_sections(self):
    #     """Extract main sections and their content."""
    #     sections = defaultdict(str)
    #     current_section = 'Introduction'
        
    #     for page in self.doc:
    #         text = page.get_text()
            
    #         # Split into lines and process each line
    #         lines = text.split('\n')
    #         for line in lines:
    #             # Check if line is a section header
    #             # if re.match(r'^[0-9\.]*\s*[A-Z][a-zA-Z\s]{2,}$', line.strip()):
    #             if self._is_section_header(line.strip()):
    #                 current_section = line.strip()
    #                 continue
                    
    #             # Add line to current section
    #             sections[current_section] += line + '\n'
                
    #     return dict(sections)
        
    def _extract_figures(self):
        """Extract figures and their captions."""
        from PIL import Image  # optional dependency, imported lazily
        figures = []
        
        for page_num, page in enumerate(self.doc):
            # Get images
            image_list = page.get_images()
            
            # Get text for caption matching
            text = page.get_text()
            
            for img_index, image in enumerate(image_list):
                try:
                    # Get image data
                    xref = image[0]
                    base_image = self.doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Convert to PIL Image
                    image_data = Image.open(io.BytesIO(image_bytes))
                    
                    # Try to find caption
                    caption = ""
                    caption_patterns = [
                        r'Figure\s+\d+[\.:]?\s*(.*?)(?=\n\n|\n[A-Z])',
                        r'Fig\.\s+\d+[\.:]?\s*(.*?)(?=\n\n|\n[A-Z])'
                    ]
                    
                    for pattern in caption_patterns:
                        caption_match = re.search(pattern, text, re.DOTALL)
                        if caption_match:
                            caption = caption_match.group(1).strip()
                            break
                    
                    figures.append({
                        'page': page_num + 1,
                        'image': image_data,
                        'caption': caption
                    })
                    
                except Exception as e:
                    print(f"Error extracting image {img_index} from page {page_num + 1}: {str(e)}")
                    
        return figures
        
    def _extract_references(self):
        """Extract references section."""
        references = []
        
        # Get text from last few pages where references typically appear
        last_pages_text = ""
        for page in self.doc[-3:]:  # Check last 3 pages
            last_pages_text += page.get_text()
            
        # Try to find references section
        ref_match = re.search(r'(References|REFERENCES)',
                              last_pages_text, re.DOTALL)
        
        if ref_match:
            # Split into individual references
            ref_text = last_pages_text[ref_match.end():].strip()
            ref_list = re.split(r'\n\s*\[\d+\]|\n\d+\.|\n(?=[A-Z])', ref_text)
            references = [ref.strip() for ref in ref_list if ref.strip()]
            
        return references
        
    def save_structure(self, output_dir):
        """Save extracted content to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract content
        # content = self.extract_structure()
        
        # Save text content as JSON
        text_content = {
            'metadata': self.content['metadata'],
            'main_text': self.content['main_text'],
            'references': self.content['references']
        }
        
        with open(os.path.join(output_dir, 'content.json'), 'w') as f:
            json.dump(text_content, f, indent=2)
            
        # Save figures
        if 'figures' not in self.content:
            return

        # Create figures directory
        figures_dir = os.path.join(output_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        
        for i, figure in enumerate(self.content['figures']):
            # Save image
            image_path = os.path.join(figures_dir, f'figure_{i+1}.png')
            figure['image'].save(image_path)
            
            # Save caption
            caption_path = os.path.join(figures_dir, f'figure_{i+1}_caption.txt')
            with open(caption_path, 'w', encoding='utf-8') as f:
                f.write(figure['caption'])
                
    def __del__(self):
        """Clean up by closing the document."""
        if hasattr(self, 'doc'):
            self.doc.close()

# Usage example
if __name__ == "__main__":
    pdf_path = "automs-final.pdf"
    output_dir = "extracted_content"
    
    extractor = PDFExtractor(pdf_path)
    content = extractor.extract_content()
    extractor.save_structure(output_dir)