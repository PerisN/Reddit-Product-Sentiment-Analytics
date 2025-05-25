#!/usr/bin/env python3

# Import libraries
import zstandard
import json
import yaml
import argparse
import logging
import os
import csv
import re
import emoji
from html import unescape
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import List, Dict, Tuple, Optional
import sys


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('reddit_search.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

class RedditSubmissionSearcher:
    """Reddit submission searcher for skincare terms and brands"""
    
    def __init__(self, config_file: str):
        """Initialize with configuration from YAML file"""
        self.load_config(config_file)
        self.setup_output_directory()
        self.init_statistics()
        self.compile_search_patterns()

    def load_config(self, config_file: str):
        """Load and validate configuration"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file {config_file} not found")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise

        # Load search terms
        self.skin_terms = self.config.get('skin_terms', [])
        self.brand_names = self.config.get('brand_names', [])
        
        if not self.skin_terms and not self.brand_names:
            logger.warning("No search terms defined in configuration")
        
        # Search options
        self.case_sensitive = self.config.get('case_sensitive', False)
        self.match_whole_words = self.config.get('match_whole_words', True)
        
        # Output options
        self.output_dir = self.config.get('output_dir') 
        self.max_selftext_length = self.config.get('max_selftext_length', 10000)

    def setup_output_directory(self):
        """Create output directory structure"""
        self.output_path = Path(self.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def init_statistics(self):
        """Initialize statistics tracking"""
        self.stats = {
            'files_processed': 0,
            'submissions_processed': 0,
            'submissions_skipped': 0,
            'matches_found': 0,
            'errors_encountered': 0,
            'matching_files': [],
            'term_counts': Counter(),
            'brand_counts': Counter(),
            'subreddit_counts': Counter(),
            'flair_counts': Counter(),
            'score_distribution': {'0-10': 0, '11-50': 0, '51-100': 0, '101-500': 0, '500+': 0}
        }

    def compile_search_patterns(self):
        """Pre-compile regex patterns for better performance"""
        self.skin_patterns = []
        self.brand_patterns = []
        
        flags = 0 if self.case_sensitive else re.IGNORECASE
        
        for term in self.skin_terms:
            if self.match_whole_words:
                pattern = re.compile(r'\b' + re.escape(term) + r'\b', flags)
            else:
                pattern = re.compile(re.escape(term), flags)
            self.skin_patterns.append((term, pattern))
        
        for brand in self.brand_names:
            if self.match_whole_words:
                pattern = re.compile(r'\b' + re.escape(brand) + r'\b', flags)
            else:
                pattern = re.compile(re.escape(brand), flags)
            self.brand_patterns.append((brand, pattern))

    def read_and_decode(self, reader, chunk_size, max_window_size, previous_chunk=None, bytes_read=0):
        """Decode chunks from ZST stream"""
        try:
            chunk = reader.read(chunk_size)
            bytes_read += chunk_size
            
            if previous_chunk is not None:
                chunk = previous_chunk + chunk
            
            try:
                return chunk.decode('utf-8')
            except UnicodeDecodeError:
                if bytes_read > max_window_size:
                    logger.warning(f"Unable to decode frame after reading {bytes_read:,} bytes")
                    return ''
                return self.read_and_decode(reader, chunk_size, max_window_size, chunk, bytes_read)
        except Exception as e:
            logger.error(f"Error reading chunk: {e}")
            return ''

    def read_lines_zst(self, file_name):
        """Generator to read lines from ZST file"""
        file_size = os.path.getsize(file_name)
        
        try:
            with open(file_name, 'rb') as file_handle:
                buffer = ''
                reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(file_handle)
                
                while True:
                    chunk = self.read_and_decode(reader, 2**27, (2**29) * 2)
                    if not chunk:
                        break
                    
                    lines = (buffer + chunk).split("\n")
                    
                    for line in lines[:-1]:
                        if line.strip():
                            progress = file_handle.tell() / file_size
                            yield line, progress
                    
                    buffer = lines[-1]
                
                reader.close()
                
        except Exception as e:
            logger.error(f"Error reading ZST file {file_name}: {e}")
            raise
    
    def clean_text(text: Optional[str]) -> str:
        """Clean and normalize text content from Reddit posts and comments."""
        if not text:
            return ""
            
        text = unescape(text) # Unescape HTML entities
        text = str(text)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text) # Remove URLs 
        text = re.sub(r'/u/\w+', '', text) # Remove Reddit user mentions
        text = re.sub(r'u/\w+', '', text) 
        text = re.sub(r'/r/\w+', '', text) # Remove subreddit references
        text = re.sub(r'r/\w+', '', text)
        text = emoji.demojize(text) # Convert emoji to text
        text = re.sub(r'[^\w\s.,!?-]', ' ', text) # Remove special characters but keep basic punctuation
        text = re.sub(r'\s+', ' ', text) # Normalize whitespace
        text = re.sub(r'([.,!?])\1+', r'\1', text) # Remove multiple punctuation
        text = text.strip() # Remove trailing whitespace
        return text

    def search_terms_in_text(self, text: str) -> Tuple[List[str], List[str]]:
        """Search for skin terms and brands using pre-compiled patterns"""
        skin_found = []
        brands_found = []
        
        for term, pattern in self.skin_patterns:
            if pattern.search(text):
                skin_found.append(term)
        
        for brand, pattern in self.brand_patterns:
            if pattern.search(text):
                brands_found.append(brand)
        
        return skin_found, brands_found


    def process_submission(self, submission_data: Dict) -> Optional[Dict]:
        """Process a single submission and return match information"""
        try:
            
            # Extract relevant fields
            title = submission_data.get('title', '')
            selftext = submission_data.get('selftext', '')

            # Truncate selftext if too long
            if len(selftext) > self.max_selftext_length:
                selftext = selftext[:self.max_selftext_length] + '...'

            # Clean title and selftext
            title_clean = self.clean_text(title)
            selftext_clean = self.clean_text(selftext)
            combined_text = f"{title_clean} {selftext_clean}"

            # Search for terms
            skin_terms, brand_terms = self.search_terms_in_text(combined_text)

            if not skin_terms and not brand_terms:
                return None

            
            # Extract fields
            subreddit = submission_data.get('subreddit', '')
            created_utc = submission_data.get('created_utc', 0)
            score = submission_data.get('score', 0)
            post_id = submission_data.get('id', '')
            url = submission_data.get('url', '')
            permalink = submission_data.get('permalink', '')
            link_flair_text = submission_data.get('link_flair_text', '')
            num_comments = submission_data.get('num_comments', 0)
            upvote_ratio = submission_data.get('upvote_ratio', 0.0)
            
            # Create Reddit link
            reddit_link = f"https://www.reddit.com{permalink}" if permalink else f"https://www.reddit.com/r/{subreddit}/comments/{post_id}/"
            
            # Format creation date
            created_date = datetime.fromtimestamp(created_utc).strftime("%Y-%m-%d %H:%M:%S")
            
            return {
                'post_id': post_id,
                'title': title_clean,
                'selftext': selftext_clean,
                'subreddit': subreddit,
                'created': created_date,
                'score': score,
                'upvote_ratio': upvote_ratio,
                'num_comments': num_comments,
                'url': url,
                'reddit_link': reddit_link,
                'link_flair_text': link_flair_text,
                'skin_terms': skin_terms,
                'brand_terms': brand_terms
            }
            
        except Exception as e:
            logger.warning(f"Error processing submission: {e}")
            self.stats['errors_encountered'] += 1
            return None

    def update_statistics(self, match: Dict):
        """Update statistics with match information"""

        for term in match['skin_terms']:
            self.stats['term_counts'][term] += 1
        for brand in match['brand_terms']:
            self.stats['brand_counts'][brand] += 1
        
        self.stats['subreddit_counts'][match['subreddit']] += 1
        if match['link_flair_text']:
            self.stats['flair_counts'][match['link_flair_text']] += 1
        
        # Score distribution
        score = match['score']
        if score <= 10:
            self.stats['score_distribution']['0-10'] += 1
        elif score <= 50:
            self.stats['score_distribution']['11-50'] += 1
        elif score <= 100:
            self.stats['score_distribution']['51-100'] += 1
        elif score <= 500:
            self.stats['score_distribution']['101-500'] += 1
        else:
            self.stats['score_distribution']['500+'] += 1


    def search_file(self, file_path: str) -> List[Dict]:
        """Search a single ZST file for matching submissions"""
        matches = []
        lines_processed = 0
        filename = Path(file_path).name
        
        logger.info(f"Processing file: {filename}")
        
        try:
            for line, progress in self.read_lines_zst(file_path):
                try:
                    submission_data = json.loads(line)
                    match = self.process_submission(submission_data)
                    
                    if match:
                        matches.append(match)
                        self.update_statistics(match)
                        
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.debug(f"Error processing line: {e}")
                    continue
                
                lines_processed += 1
                self.stats['submissions_processed'] += 1
                
                # Progress logging
                if lines_processed % 50000 == 0:
                    logger.info(f"Progress: {lines_processed:,} lines ({progress:.1%}) - {len(matches)} matches found")
                    
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            self.stats['errors_encountered'] += 1
            return matches
        
        if matches:
            self.stats['matching_files'].append(filename)
            logger.info(f"Found {len(matches)} matches in {filename}")
        else:
            logger.info(f"No matches found in {filename}")
        
        return matches

    def save_matches_to_csv(self, matches: List[Dict], base_filename: str):
        """Save matches to CSV file"""
        if not matches:
            return
            
        csv_filename = f"{base_filename}_matches.csv"
        output_path = self.output_path / csv_filename
        
        # Define CSV columns
        fieldnames = [
            'post_id', 'title', 'selftext', 'subreddit', 'created',
            'score', 'upvote_ratio', 'num_comments', 'url', 'reddit_link',
            'link_flair_text', 'skin_terms', 'brand_terms'
        ]
        
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for match in matches:
                    row = {
                        'post_id': match['post_id'],
                        'title': match['title'].replace('\n', ' ').replace('\r', ' '),
                        'selftext': match['selftext'].replace('\n', ' ').replace('\r', ' '),
                        'subreddit': match['subreddit'],
                        'created': match['created'],
                        'score': match['score'],
                        'upvote_ratio': match['upvote_ratio'],
                        'num_comments': match['num_comments'],
                        'url': match['url'],
                        'reddit_link': match['reddit_link'],
                        'link_flair_text': match['link_flair_text'],
                        'skin_terms': '; '.join(match['skin_terms']),
                        'brand_terms': '; '.join(match['brand_terms'])
                    }
                    writer.writerow(row)
            
            logger.info(f"Saved {len(matches)} matches to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving CSV file {output_path}: {e}")

    def save_statistics(self):
        """Save search statistics"""
        stats_path = self.output_path / "search_statistics.json"
        
        try:
            stats_to_save = {
                'search_summary': {
                    'files_processed': self.stats['files_processed'],
                    'submissions_processed': self.stats['submissions_processed'],
                    'submissions_skipped': self.stats['submissions_skipped'],
                    'matches_found': self.stats['matches_found'],
                    'errors_encountered': self.stats['errors_encountered'],
                    'matching_files': self.stats['matching_files']
                },
                'term_statistics': {
                    'top_skin_terms': dict(self.stats['term_counts'].most_common(50)),
                    'top_brands': dict(self.stats['brand_counts'].most_common(50)),
                    'total_unique_skin_terms': len(self.stats['term_counts']),
                    'total_unique_brands': len(self.stats['brand_counts'])
                },
                'subreddit_statistics': {
                    'top_subreddits': dict(self.stats['subreddit_counts'].most_common(30)),
                    'total_subreddits': len(self.stats['subreddit_counts'])
                },
                'content_statistics': {
                    'top_flairs': dict(self.stats['flair_counts'].most_common(30)),
                    'score_distribution': self.stats['score_distribution']
                }
            }
            
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats_to_save, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Statistics saved to {stats_path}")
            
        except Exception as e:
            logger.error(f"Error saving statistics: {e}")

    def print_summary(self):
        """Print search summary"""
        logger.info("\n" + "="*60)
        logger.info("SEARCH COMPLETE - SUMMARY")
        logger.info("="*60)
        logger.info(f"Files processed: {self.stats['files_processed']}")
        logger.info(f"Submissions processed: {self.stats['submissions_processed']:,}")
        logger.info(f"Submissions skipped: {self.stats['submissions_skipped']:,}")
        logger.info(f"Matches found: {self.stats['matches_found']:,}")
        logger.info(f"Files with matches: {len(self.stats['matching_files'])}")
        logger.info(f"Errors encountered: {self.stats['errors_encountered']}")
        
        if self.stats['matches_found'] > 0:
            logger.info(f"\nTop 10 skin terms:")
            for term, count in self.stats['term_counts'].most_common(10):
                logger.info(f"  {term}: {count:,}")
            
            logger.info(f"\nTop 10 brands:")
            for brand, count in self.stats['brand_counts'].most_common(10):
                logger.info(f"  {brand}: {count:,}")
            
            logger.info(f"\nTop 10 subreddits:")
            for subreddit, count in self.stats['subreddit_counts'].most_common(10):
                logger.info(f"  r/{subreddit}: {count:,}")
            
            logger.info(f"\nScore distribution:")
            for range_key, count in self.stats['score_distribution'].items():
                logger.info(f"  {range_key}: {count:,}")
        
        logger.info("="*60)

    def search_directory(self, directory: str):
        """Search all Reddit submission files in directory"""
        directory_path = Path(directory)
        
        if not directory_path.exists():
            logger.error(f"Directory {directory} does not exist")
            return
        
        # Find all .zst files that look like Reddit submissions
        zst_files = []
        for file_path in directory_path.rglob('*.zst'):
            if file_path.is_file():
                filename = file_path.name.lower()
                if any(indicator in filename for indicator in ['RS_']):
                    zst_files.append(file_path)
        
        if not zst_files:
            logger.warning(f"No Reddit submission files found in {directory}")
            return
        
        logger.info(f"Found {len(zst_files)} Reddit submission files")
        
        # Process each file and save matches to CSV
        all_matches = []
        for file_path in sorted(zst_files):
            try:
                matches = self.search_file(str(file_path))
                self.stats['files_processed'] += 1
                self.stats['matches_found'] += len(matches)
                
                base_filename = file_path.stem
                self.save_matches_to_csv(matches, base_filename)
                all_matches.extend(matches)
                
            except Exception as e:
                logger.error(f"Failed to process file {file_path}: {e}")
                continue
        
        # Save combined results, statistics and print summary
        if all_matches:
            logger.info(f"Saving combined results with {len(all_matches)} total matches")
            self.save_matches_to_csv(all_matches, "all_combined")
        
        self.save_statistics()
        self.print_summary()

def main():
    parser = argparse.ArgumentParser(description='Search Reddit submissions for skincare terms and brands')
    parser.add_argument('directory', help='Directory containing Reddit submission .zst files')
    parser.add_argument('--config', '-c', default='config.yaml')
    
    args = parser.parse_args()
    
    if not Path(args.config).exists():
        logger.error(f"Configuration file {args.config} not found")
        return
    
    searcher = RedditSubmissionSearcher(args.config)
    searcher.search_directory(args.directory)


if __name__ == "__main__":
    main()