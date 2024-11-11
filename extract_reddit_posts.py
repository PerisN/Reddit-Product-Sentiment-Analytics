# Import libraries
import os
import sys
import re
import time
import logging
import yaml
import praw
import pandas as pd
from dotenv import load_dotenv
from html import unescape
import emoji
from datetime import datetime, timedelta
from functools import lru_cache
from typing import List, Dict, Any, Optional, Iterator, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


"""
Reddit Data Collection and Processing Script
--------------------------------------------
Purpose:
    This script collects and processes posts and comments from specified subreddits on Reddit.
    It focuses on gathering data related to specific search terms, extracting relevant content
    from both posts and comments, and saving the results in CSV format for further analysis.

Main Features:
    - Collects posts and comments from multiple subreddits based on user-defined search terms
    - Filters and cleans the collected data using regex to remove unwanted characters
    - Exports collected data into CSV files with relevant timestamps
    - Implements error handling and logging for tracking script progress and identifying issues
    - Supports multi-threaded execution to optimize data collection

Required Environment Variables:
    - CLIENT_ID: Reddit API client ID
    - CLIENT_SECRET: Reddit API client secret
    - USER_AGENT: Reddit API user agent string

Configuration (via config.yaml):
    - subreddits: List of subreddit names to collect data from
    - search_terms: List of search terms to filter posts and comments by
    - post_limit: Number of posts to include in the final dataset
    - comment_limit: Number of top comments to collect for each post
    - output_dir: Directory where the CSV files will be saved
    - logging_level: Log level for tracking events (e.g., DEBUG, INFO, WARNING)
"""

# Initialize logger
logger = logging.getLogger(__name__)
csv_lock = Lock()

@lru_cache(maxsize=None)
def load_config(config_file: str = 'config.yaml') -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
            return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise

def setup_logging(config: dict) -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format=config['logging']['format']
    )
    return logging.getLogger(__name__)

def initialize_reddit() -> praw.Reddit:
    """Initialize Reddit API connection."""
    load_dotenv()
    
    try:
        reddit = praw.Reddit(
            client_id=os.getenv('CLIENT_ID'),
            client_secret=os.getenv('CLIENT_SECRET'),
            user_agent=os.getenv('USER_AGENT')
        )
        reddit.user.me()  # Verify authentication
        return reddit
    except Exception as e:
        logger.error(f"Reddit authentication failed: {e}")
        raise

def compile_term_patterns(search_term: str) -> re.Pattern:
    """Compile regex patterns for the search term."""
    variations = [
        search_term,
        search_term.lower(),
        search_term.replace(' ', '-'),
        search_term.replace(' ', '_')
    ]
    pattern = '|'.join(map(re.escape, variations))
    return re.compile(pattern, re.IGNORECASE)

def contains_term(text: str, term_patterns: re.Pattern) -> bool:
    """Check if text contains specified term using precompiled regex."""
    if not text:
        return False
    return bool(term_patterns.search(text))

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


def check_media_type(submission: praw.models.Submission) -> Dict[str, Any]:
    """Determine the media type of a submission."""
    media_info = {
        'has_image': False,
        'has_video': False,
        'media_count': 0
    }
    
    try:
        if submission.is_video:
            media_info['has_video'] = True
            media_info['media_count'] += 1
        
        if hasattr(submission, 'preview'):
            media_info['has_image'] = True
            if 'images' in submission.preview:
                media_info['media_count'] += len(submission.preview['images'])
        
        if hasattr(submission, 'is_gallery') and submission.is_gallery:
            media_info['has_image'] = True
            if hasattr(submission, 'gallery_data'):
                media_info['media_count'] += len(submission.gallery_data['items'])
        
        if submission.url.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
            media_info['has_image'] = True
            if media_info['media_count'] == 0:
                media_info['media_count'] += 1
            
    except Exception as e:
        logger.warning(f"Error checking media type for submission {submission.id}: {e}")
    
    return media_info

def get_posts(subreddit: praw.models.Subreddit, config: dict, position: int) -> Iterator[praw.models.Submission]:
    """Fetch posts using time windows with timeout and progress tracking."""
    months = config.get('months')
    post_limit = config.get('post_limit')
    search_term = config['search_term']
    term_patterns = compile_term_patterns(search_term)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30.44 * months)
    window_size = timedelta(days=14)
    current_window_end = end_date
    
    posts_processed = 0
    relevant_posts = 0
    last_update_time = time.time()
    stall_timeout = 300  # 5 minutes timeout
    
    pbar = tqdm(
        desc=f"Processing r/{subreddit.display_name}",
        unit=" posts",
        position=position + 1,
        leave=False,
        ncols=100
    )

    while current_window_end > start_date and relevant_posts < post_limit:
        current_window_start = max(current_window_end - window_size, start_date)
        
        try:
            # Check for timeout
            if time.time() - last_update_time > stall_timeout:
                logger.warning(f"Timeout reached for r/{subreddit.display_name}")
                break
                
            for post in subreddit.new(limit=1000):  # Limit each window to 1000 posts
                posts_processed += 1
                last_update_time = time.time()  # Reset timeout counter
                
                post_date = datetime.fromtimestamp(post.created_utc)
                
                if post_date > current_window_end:
                    continue
                if post_date < current_window_start:
                    break
                
                if (contains_term(post.title, term_patterns) or 
                    contains_term(post.selftext, term_patterns)):
                    relevant_posts += 1
                    pbar.set_description(f"r/{subreddit.display_name} [{relevant_posts} found]")
                    pbar.update(1)
                    yield post
                    
                    if relevant_posts >= post_limit:
                        break
                
                time.sleep(0.1)
            
            current_window_end = current_window_start
            
        except Exception as e:
            logger.warning(f"Error in window {current_window_start} to {current_window_end}: {e}")
            time.sleep(5)
            continue
            
    pbar.close()
    print(f"\r\033[K", end="")
    logger.info(f"Processed {posts_processed:,} posts, found {relevant_posts:,} relevant posts in r/{subreddit.display_name}")


def process_comments(submission: praw.models.Submission, term_patterns: re.Pattern, config: dict) -> Tuple[List[Dict], bool]:
    """Process comments from a submission."""
    comments = []
    term_found = False
    comment_limit = config.get('comment_limit')

    try:
        submission.comments.replace_more(limit=0)
        valid_comments = [
            comment for comment in submission.comments.list()
            if (hasattr(comment, 'author') and 
                comment.author is not None and 
                comment.author.name != 'AutoModerator')
        ][:comment_limit]
        
        for comment in valid_comments:
            contains_search_term = contains_term(comment.body, term_patterns)
            
            if contains_search_term:
                term_found = True
                cleaned_body = clean_text(comment.body)
                
                comment_data = {
                    'post_id': submission.id,
                    'body': cleaned_body,
                    'score': comment.score,
                    'contains_term': True
                }
                comments.append(comment_data)
            
        return comments, term_found
        
    except Exception as e:
        logger.warning(f"Error processing comments for submission {submission.id}: {e}")
        return [], False
        

def process_submission(submission: praw.models.Submission, search_term: str, config: dict) -> Optional[Tuple[Dict, List[Dict]]]:
    """Process a single submission and its comments."""
    try:
        term_patterns = compile_term_patterns(search_term)
        
        title_contains = contains_term(submission.title, term_patterns)
        body_contains = contains_term(submission.selftext, term_patterns)
        
        comments, comments_contain = process_comments(submission, term_patterns, config)
        
        if not (title_contains or body_contains or comments_contain):
            return None
            
        cleaned_title = clean_text(submission.title)
        cleaned_selftext = clean_text(submission.selftext)
            
        media_info = check_media_type(submission)
        
        post_data = {
            'post_id': submission.id,
            'subreddit': submission.subreddit.display_name,
            'flair': submission.link_flair_text,
            'title': cleaned_title,
            'selftext': cleaned_selftext,
            'score': submission.score,
            'num_comments': submission.num_comments,
            'created_utc': datetime.fromtimestamp(submission.created_utc),
            'has_image': media_info['has_image'],
            'has_video': media_info['has_video'],
            'media_count': media_info['media_count'],
            'permalink': f"https://reddit.com{submission.permalink}"
        }

        return post_data, comments

    except Exception as e:
        logger.warning(f"Error processing submission {submission.id}: {e}")
        return None


def process_subreddit(subreddit_name: str, reddit: praw.Reddit, config: dict, position: int) -> Tuple[List[dict], List[dict]]:
    """Process a single subreddit with timeout."""
    subreddit = reddit.subreddit(subreddit_name)
    subreddit_posts, subreddit_comments = [], []
    
    try:
        for submission in get_posts(subreddit, config, position):
            result = process_submission(submission, config['search_term'], config)
            if result:
                post_data, comments = result
                subreddit_posts.append(post_data)
                subreddit_comments.extend(comments)
                
        return subreddit_posts, subreddit_comments
        
    except Exception as e:
        logger.error(f"Error processing r/{subreddit_name}: {e}")
        return [], []


def save_to_csv(posts: List[Dict], comments: List[Dict], output_folder: str, search_term: str) -> None:
    """Save a batch of posts and comments to CSV files with thread safety."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with csv_lock:
        os.makedirs(output_folder, exist_ok=True)
        pd.DataFrame(posts).to_csv(f"{output_folder}/reddit_{search_term}_posts_{timestamp}.csv", index=False)
        if comments:
            pd.DataFrame(comments).to_csv(f"{output_folder}/reddit_{search_term}_comments_{timestamp}.csv", index=False)
        logger.info(f"Saved posts and comments to {output_folder}")


def collect_data(config: dict, reddit: praw.Reddit) -> None:
    """Main collection function."""
    all_posts = []
    all_comments = []
    start_time = datetime.now()
    
    print("\033[2J\033[H")
    
    main_pbar = tqdm(
        total=len(config['subreddits']),
        desc="Overall Progress",
        position=0,
        leave=True,
        ncols=100
    )

    max_workers = min(len(config['subreddits']), 6)
    timeout_per_subreddit = 600  # 10 minutes timeout per subreddit
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_subreddit = {
            executor.submit(
                process_subreddit, 
                subreddit, 
                reddit, 
                config,
                idx
            ): subreddit
            for idx, subreddit in enumerate(config['subreddits'])
        }
        
        try:
            for future in as_completed(future_to_subreddit):
                subreddit = future_to_subreddit[future]
                try:
                    # Add timeout to future.result()
                    subreddit_posts, subreddit_comments = future.result(timeout=timeout_per_subreddit)
                    if subreddit_posts:
                        all_posts.extend(subreddit_posts)
                        all_comments.extend(subreddit_comments)
                    main_pbar.update(1)
                except TimeoutError:
                    logger.error(f"Timeout processing r/{subreddit}")
                    main_pbar.update(1)
                except Exception as e:
                    logger.error(f"Error processing r/{subreddit}: {e}")
                    main_pbar.update(1)
        finally:
            main_pbar.close()
            
    print("\n" * 2)
    
    if all_posts:
        save_to_csv(all_posts, all_comments, config['output_folder'], config['search_term'])

    elapsed = datetime.now() - start_time
    logger.info("\nCollection Summary:")
    logger.info(f"Total posts collected: {len(all_posts):,}")
    logger.info(f"Total comments collected: {len(all_comments):,}")
    logger.info(f"Time elapsed: {elapsed}")

def main():
    """Entry point"""
    try:
            
        config = load_config()
        logger = setup_logging(config)
        
        # Clear screen and position cursor at top
        print("\033[2J\033[H")
        
        logger.info("Starting Reddit data collection...")
        logger.info(f"Config loaded: {config}")
        
        reddit = initialize_reddit()
        logger.info("Reddit API initialized successfully.")
        
        print("\n")
        
        collect_data(config, reddit)

    except KeyboardInterrupt:
        print("\n" * 10) 
        logger.info("\nScript interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print("\n" * 10) 
        logger.error(f"Application error: {e}")
        raise


if __name__ == "__main__":
    main()