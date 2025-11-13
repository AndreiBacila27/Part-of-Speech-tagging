import os
import re
import json
from collections import defaultdict
from typing import Dict, DefaultDict

class BrownCorpusAnalyzer:
    def __init__(self, corpus_dir: str = 'brown'):
        self.corpus_dir = corpus_dir
        # Structure: {word: {pos: count}}
        self.word_pos_counts: DefaultDict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.pos_total_counts: DefaultDict[str, int] = defaultdict(int)
        # Only include files starting with 'c' followed by a letter and two digits
        self.file_pattern = re.compile(r"^c[a-z]\d{2}$")
        
    def clean_pos_tag(self, tag: str) -> str:
        """
        Clean POS tag following the rules from section 1.2 of the PDF:
        1. Remove -HL, -TL, -NC suffixes
        2. Remove all FW-, NC-, NP- prefixes
        3. Keep first part before any +
        4. Convert to lowercase
        """
        if not tag or tag == 'nil':
            return 'nil'

        tag = tag.lower()
        
        # Skip punctuation tags
        if tag in ['.', ',', ':', ';', '(', ')', '"', "'", '?', '!']:
            return None
            
        # Remove -HL, -TL, -NC suffixes
        while re.search(r'-(hl|tl|nc)(?:-|$)', tag):
            tag = re.sub(r'-(hl|tl|nc)(?:-|$)', '', tag)
        
        # Remove FW-, NC-, NP- prefixes
        if any(tag.startswith(prefix) for prefix in ['fw-', 'nc-', 'np-']):
            tag = tag[3:]
            
        # For tags with '+', keep only the part before '+'
        if '+' in tag:
            tag = tag.split('+')[0]
            
        # Remove any remaining hyphens
        if '-' in tag:
            tag = tag.split('-')[0]
            
        return tag.strip() or 'nil'
    
    def clean_word(self, word: str) -> str:
        """Clean word following section 1.3 rules:
        - Remove leading/trailing quotes and spaces
        - Handle possessives
        - Keep hyphenated and numeric forms intact
        """
        # Remove leading/trailing quotes and spaces
        word = word.strip('\'" ')
        
        # Handle possessive forms (word's -> word)
        word = re.sub(r"'s$", "", word)
        
        # Keep hyphenated words and numeric forms intact
        if '-' in word:
            # Keep numeric ranges (1-2, 1940-50)
            if re.match(r'^\d+(?:-\d+)+$', word):
                return word
            # Keep hyphenated compounds unless they end in hyphen
            if not word.endswith('-'):
                return word
        
        # Keep numeric forms with slashes (1/2, 3/4)
        if re.match(r'^\d+/\d+$', word):
            return word
            
        return word.strip()
    
    def process_tuple(self, tuple: str) -> None:
        """Process a single tuple of format word/POS"""
        if not tuple or '/' not in tuple:
            return
            
        try:
            # Handle special cases where the last '/' isn't the POS separator
            if re.match(r'^\d+(?:/\d+)+/[a-zA-Z]+$', tuple):
                # For cases like "1/2/cd", keep "1/2" as word
                parts = tuple.split('/')
                word = '/'.join(parts[:-1])
                pos = parts[-1]
            else:
                # Normal case: split on last '/'
                word, pos = tuple.rsplit('/', 1)
                
            # Skip empty words or POS tags
            if not word or not pos:
                return
                
            # Clean the word and POS tag
            cleaned_word = self.clean_word(word)
            cleaned_pos = self.clean_pos_tag(pos)
            
            # Skip punctuation and only process if both word and POS are valid
            if cleaned_word and cleaned_pos is not None:
                self.word_pos_counts[cleaned_word][cleaned_pos] += 1
                self.pos_total_counts[cleaned_pos] += 1
                    
        except Exception as e:
            print(f"Error processing tuple '{tuple}': {e}")
    
    def process_file_content(self, content: str) -> None:
        """Process all tuples in a file's content"""
        for tuple in re.split(r"\s+", content.strip()):
            self.process_tuple(tuple)
    
    def read_corpus_file(self, file_path: str) -> None:
        """Read and process a single corpus file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.process_file_content(content)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    def analyze_corpus(self) -> None:
        """Analyze all matching files in the corpus directory"""
        try:
            all_files = os.listdir(self.corpus_dir)
        except FileNotFoundError:
            print(f"Error: Directory '{self.corpus_dir}' not found")
            return
        
        # Only process files matching the pattern c[a-z]\d{2}
        matching_files = [f for f in all_files if self.file_pattern.match(f)]
        total_files = len(matching_files)
        print(f"Processing {total_files} files...")
        
        for i, file in enumerate(matching_files, 1):
            self.read_corpus_file(os.path.join(self.corpus_dir, file))
            
    def save_dictionaries(self) -> None:
        """Save dictionaries to JSON files for analysis"""
        try:
            # Save complete word-POS counts (including special characters/numbers)
            word_pos_dict = {word: dict(pos_counts) for word, pos_counts in self.word_pos_counts.items()}
            
            with open('word_pos_counts.json', 'w', encoding='utf-8') as f:
                json.dump(word_pos_dict, f, indent=2, sort_keys=True)
            print("Saved complete word-POS counts to word_pos_counts.json")
            
            # Save words-only dictionary (filtering out special characters and numbers)
            words_only_dict = {
                word: dict(pos_counts)
                for word, pos_counts in self.word_pos_counts.items()
                if word.isalpha()  # Keep only pure alphabetic words
            }
            
            with open('words_only_counts.json', 'w', encoding='utf-8') as f:
                json.dump(words_only_dict, f, indent=2, sort_keys=True)
            print("Saved words-only counts to words_only_counts.json")
            
            # Save POS total counts
            with open('pos_total_counts.json', 'w', encoding='utf-8') as f:
                json.dump(dict(self.pos_total_counts), f, indent=2, sort_keys=True)
            print("Saved POS total counts to pos_total_counts.json")
            
        except Exception as e:
            print(f"Error saving dictionaries: {e}")
            
def main():
    # Initialize and run the analyzer
    analyzer = BrownCorpusAnalyzer()
    print("Analyzing Brown Corpus...")
    analyzer.analyze_corpus()
    
    # Save dictionaries to JSON files
    analyzer.save_dictionaries()
    
    # Print corpus statistics
    print("\nCorpus Statistics:")
    
    # Count pure words only (no numbers or special characters)
    pure_words = {word: counts for word, counts in analyzer.word_pos_counts.items() if word.isalpha()}
    pure_word_occurrences = sum(sum(pos_counts.values()) for pos_counts in pure_words.values())
    
    print(f"Total number of words (including repetitions):")
    print(f"  - All tuples: {sum(analyzer.pos_total_counts.values()):,d}")
    print(f"  - Pure words only: {pure_word_occurrences:,d}")
    
    print(f"\nTotal number of distinct words:")
    print(f"  - All tuples: {len(analyzer.word_pos_counts):,d}")
    print(f"  - Pure words only: {len(pure_words):,d}")
    
    print(f"\nTotal number of distinct parts of speech: {len(analyzer.pos_total_counts)}")
    
    print("\nOccurrences for each Part of Speech:")
    # Sort POS tags by frequency in descending order
    sorted_pos = sorted(analyzer.pos_total_counts.items(), key=lambda x: x[1], reverse=True)
    for pos, count in sorted_pos:
        print(f"{pos}: {count:,d}")

if __name__ == '__main__':
    main()