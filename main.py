import os, re, json, string, requests
from collections import defaultdict
from typing import Dict, DefaultDict


class BrownCorpusAnalyzer:
    def __init__(self, corpus_dir: str = 'brown', stopwords_file: str = 'stopwords.txt'):
        """
        Initializes the Brown Corpus analyzer and sets up the tag grouping rules.

        This method configures the corpus directory, initializes dictionaries for tracking
        word and POS tag frequencies, loads the stop words list, and defines the mapping
        from detailed POS tags to simplified categories (e.g., mapping 'bed', 'vbz' to 'VERB').
        """
        self.corpus_dir = corpus_dir
        self.word_pos_counts: DefaultDict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.pos_total_counts: DefaultDict[str, int] = defaultdict(int)
        self.grouped_pos_counts: DefaultDict[str, int] = defaultdict(int)
        self.file_pattern = re.compile(r"^c[a-z]\d{2}$")
        self.stopwords = [word for word in open(stopwords_file, 'r', encoding='utf-8').read().splitlines() if word]
        # print(self.stopwords)

        self.pos_groups = {
            'NOUN': ['nn', 'nns', 'np', 'nps', 'nr', 'n$', 'np$', 'nns$', 'nr$', 'nrs'],
            'VERB': ['vb', 'vbd', 'vbg', 'vbn', 'vbz', 'be', 'bed', 'bedz', 'beg', 'bem',
                     'ben', 'ber', 'bez', 'do', 'dod', 'doz', 'hv', 'hvd', 'hvg', 'hvn', 'hvz',
                     'md', 'md*', 'do*'],
            'ADJ': ['jj', 'jjr', 'jjs', 'jjt'],
            'ADV': ['rb', 'rbr', 'rbt', 'rn', 'ql', 'qls', 'qlp', 'wql'],
            'PRON': ['pp$', 'pp$$', 'ppl', 'ppls', 'ppo', 'pps', 'ppss', 'wps', 'wp$', 'wpo'],
            'DET': ['at', 'dt', 'dti', 'dts', 'dtx', 'wdt'],
            'ADP': ['in', 'to'],
            'CONJ': ['cc', 'cs'],
            'NUM': ['cd', 'od'],
        }

        self.tag_to_group = {}
        for group, tags in self.pos_groups.items():
            for tag in tags:
                self.tag_to_group[tag] = group

    def get_pos_group(self, tag: str) -> str:
        """
        Maps a specific POS tag to a broader, simplified category.

        This function checks if the given tag belongs to one of the predefined groups
        (like NOUN, VERB, ADJ). If the tag is not found in the mapping or is 'nil',
        it returns 'OTHER'.
        """
        if tag == 'nil':
            return 'OTHER'
        return self.tag_to_group.get(tag, 'OTHER')

    def clean_pos_tag(self, tag: str) -> str:
        """
        Normalizes and cleans a raw POS tag string.

        This handles various tag modifiers found in the corpus:
        1. Simplifies compound tags (e.g., removing parts after '+').
        2. Removes prefixes for foreign words ('fw-') or cited words ('nc-', 'np-').
        3. Strips suffixes related to headlines or titles ('-hl', '-tl').
        4. Returns 'nil' if the tag is empty or invalid.
        """
        if not tag or tag == 'nil':
            return 'nil'
        if '*' in tag:
            return
        while re.search(r'-(hl|tl|nc)(?:-|$)', tag):
            tag = re.sub(r'-(hl|tl|nc)(?:-|$)', '', tag)
        if any(tag.startswith(prefix) for prefix in ['fw-', 'nc-', 'np-']):
            tag = tag[3:]
        if '+' in tag:
            tag = tag.split('+')[0]
        if '-' in tag:
            tag = tag.split('-')[0]

        return tag.strip() or 'nil'

    def process_compound_word(self, word: str, pos: str) -> None:
        """
        Handles complex word tokens that contain forward slashes.

        If a token contains a slash but is not a simple fraction or date (e.g., 'word1/word2'),
        this function splits it into component parts and updates the statistics for each part
        individually. Returns True if the word was processed as a compound.
        """
        if '/' in word and not re.match(r'^\d+/\d+$', word):
            parts = word.split('/')
            for part in parts:
                cleaned_word = self.clean_word(part)
                if cleaned_word and pos is not None:
                    self.word_pos_counts[cleaned_word][pos] += 1
                    self.pos_total_counts[pos] += 1
            return True
        return False

    def clean_word(self, word: str) -> str:
        """
        Sanitizes and normalizes a word string.

        This function performs several cleaning steps:
        - Strips surrounding quotes and spaces.
        - Removes possessive endings ("'s").
        - Filters out tokens that consist solely of punctuation.
        - Preserves valid hyphenated words (e.g., "1940-50") and fractions.
        Returns None if the word should be discarded.
        """
        word = word.strip('\'" ')

        word = re.sub(r"'s$", "", word)

        # if word in self.stopwords:
        #     return None

        if word and all(char in string.punctuation for char in word):
            return None

        if '-' in word:
            # Keep e.g., 1-2, 1940-50
            if re.match(r'^\d+(?:-\d+)+$', word):
                return word
            if not word.endswith('-'):
                return word
        # Keep e.g., 1/2, 3/4
        if re.match(r'^\d+/\d+$', word):
            return word

        return word if word else None

    def process_tuple(self, tuple: str) -> None:
        """
        Parses and processes a single 'word/tag' tuple from the corpus.

        This function handles the extraction of the word and its tag from the raw string.
        It accounts for complex cases involving multiple slashes (e.g., 'and/or/cc') and
        updates the global frequency counters for words, specific POS tags, and grouped POS categories.
        """
        if not tuple or '/' not in tuple:
            return
        try:
            # Handle specific cases and/or/cc, input/output/nn, origin/destination/nn
            if re.match(r'^([a-z]+)/([a-z]+)/([a-z]+)$', tuple):
                parts = tuple.split('/')
                pos = parts[-1]
                word1, word2 = parts[0], parts[1]

                for word in [word1, word2]:
                    cleaned_word = self.clean_word(word)
                    cleaned_pos = self.clean_pos_tag(pos)
                    if cleaned_word and cleaned_pos is not None:
                        self.word_pos_counts[cleaned_word][cleaned_pos] += 1
                        self.pos_total_counts[cleaned_pos] += 1

                        pos_group = self.get_pos_group(cleaned_pos)
                        self.grouped_pos_counts[pos_group] += 1
                return

            # Handle cases like 1/2/cd
            elif re.match(r'^\d+(?:/\d+)+/[a-z]+$', tuple):
                parts = tuple.split('/')
                word = '/'.join(parts[:-1])
                pos = parts[-1]
            else:
                word, pos = tuple.rsplit('/', 1)
            if not word or not pos:
                return

            cleaned_pos = self.clean_pos_tag(pos)
            if cleaned_pos is None:
                return

            if not self.process_compound_word(word, cleaned_pos):
                cleaned_word = self.clean_word(word)
                if cleaned_word is not None:
                    self.word_pos_counts[cleaned_word][cleaned_pos] += 1
                    self.pos_total_counts[cleaned_pos] += 1

                    pos_group = self.get_pos_group(cleaned_pos)
                    self.grouped_pos_counts[pos_group] += 1
        except Exception as e:
            print(f"Error processing tuple '{tuple}': {e}")

    def process_file_content(self, content: str) -> None:
        """
        Iterates through the raw content of a file and processes each token.

        Splits the file content by whitespace into individual tuples and delegates
        the processing of each tuple to the process_tuple method.
        """
        content = content.lower()
        for tuple in re.split(r"\s+", content.strip()):
            self.process_tuple(tuple)

    def read_corpus_file(self, file_path: str) -> None:
        """
        Reads a single file from the corpus directory.

        Opens the file in read mode with UTF-8 encoding and passes its content
        to be processed. Catches and logs any IO errors encountered.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.process_file_content(content)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    def analyze_corpus(self) -> None:
        """
        Orchestrates the analysis of the entire corpus directory.

        Iterates through all files in the configured directory that match the
        expected filename pattern and processes them sequentially.
        """
        try:
            all_files = os.listdir(self.corpus_dir)
        except FileNotFoundError:
            print(f"Error: Directory '{self.corpus_dir}' not found")
            return
        matching_files = [f for f in all_files if self.file_pattern.match(f)]
        total_files = len(matching_files)
        print(f"Processing {total_files} files...")
        for i, file in enumerate(matching_files, 1):
            self.read_corpus_file(os.path.join(self.corpus_dir, file))

    def save_dictionaries(self) -> None:
        """
        Saves the computed statistics to JSON files.

        Exports the following data structures to disk:
        - 'word_pos_counts.json': Counts of each POS tag for every word.
        - 'words_only_counts.json': Same as above but filtered for alphabetic words only.
        - 'pos_total_counts.json': Total occurrence count for each specific POS tag.
        - 'pos_grouped_counts.json': Total occurrence count for each simplified POS group.
        """
        try:
            word_pos_dict = {word: dict(pos_counts) for word, pos_counts in self.word_pos_counts.items()}
            with open('word_pos_counts.json', 'w', encoding='utf-8') as f:
                json.dump(word_pos_dict, f, indent=2, sort_keys=True)
            print("Saved complete word-POS counts to word_pos_counts.json")
            words_only_dict = {
                word: dict(pos_counts)
                for word, pos_counts in self.word_pos_counts.items()
                if word.isalpha()
            }
            with open('words_only_counts.json', 'w', encoding='utf-8') as f:
                json.dump(words_only_dict, f, indent=2, sort_keys=True)
            print("Saved words-only counts to words_only_counts.json")
            with open('pos_total_counts.json', 'w', encoding='utf-8') as f:
                json.dump(dict(self.pos_total_counts), f, indent=2, sort_keys=True)
            print("Saved POS total counts to pos_total_counts.json")

            with open('pos_grouped_counts.json', 'w', encoding='utf-8') as f:
                json.dump(dict(self.grouped_pos_counts), f, indent=2, sort_keys=True)
            print("Saved grouped POS counts to pos_grouped_counts.json")
        except Exception as e:
            print(f"Error saving dictionaries: {e}")


def main():
    """
    Main entry point for the script.

    Initializes the analyzer, runs the corpus analysis, saves the results, and
    prints a statistical summary to the console, including total word counts,
    unique word counts, and distribution of POS tags.
    """
    analyzer = BrownCorpusAnalyzer()
    print("Analyzing Brown Corpus...")
    analyzer.analyze_corpus()
    analyzer.save_dictionaries()
    print("\nCorpus Statistics:")
    pure_words = {word: counts for word, counts in analyzer.word_pos_counts.items() if word.isalpha()}
    pure_word_occurrences = sum(sum(pos_counts.values()) for pos_counts in pure_words.values())
    print(f"Total number of words (including repetitions):")
    print(f"  - All tuples: {sum(analyzer.pos_total_counts.values()):,d}")
    print(f"  - Pure words only: {pure_word_occurrences:,d}")
    print(f"\nTotal number of distinct words:")
    print(f"  - All tuples: {len(analyzer.word_pos_counts):,d}")
    print(f"  - Pure words only: {len(pure_words):,d}")
    print(f"\nTotal number of distinct parts of speech: {len(analyzer.pos_total_counts)}")
    print(f"\nNumber of Parts of Speech: {len(analyzer.pos_total_counts)}")
    print("\nOccurrences for each Part of Speech:")
    sorted_pos = sorted(analyzer.pos_total_counts.items(), key=lambda x: x[1], reverse=True)
    for pos, count in sorted_pos:
        print(f"{pos}: {count:,d}")

    print("\n" + "=" * 50)
    print("\nGrouped POS Statistics:")
    print(f"Number of POS Groups: {len(analyzer.grouped_pos_counts)}")
    print("\nOccurrences for each POS Group:")
    sorted_groups = sorted(analyzer.grouped_pos_counts.items(), key=lambda x: x[1], reverse=True)
    for group, count in sorted_groups:
        print(f"{group}: {count:,d}")


if __name__ == '__main__':
    main()