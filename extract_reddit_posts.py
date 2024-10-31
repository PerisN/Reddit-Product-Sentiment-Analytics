import os
import re
import time
import logging
import yaml
import praw
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta
from functools import lru_cache
from typing import List, Dict, Any, Optional, Iterator, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Initialize logger at module level
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

def get_top_posts(subreddit: praw.models.Subreddit, config: dict) -> Iterator[praw.models.Submission]:
    """Get top posts using efficient pagination with progress bar and month filtering."""
    try:
        params = {'t': 'all'}  # Use 'all' to get more posts
        total_fetched = 0
        posts_yielded = 0
        cutoff_date = None

        months = config.get('months')
        if months:
            cutoff_date = datetime.now() - timedelta(days=30.44 * months)
            
        pbar = tqdm(desc=f"Fetching posts from r/{subreddit.display_name}", unit=" posts", ncols=100)
        
        while True:
            try:
                new_posts = list(subreddit.new(limit=100, params=params))
                
                if not new_posts:
                    break
                    
                total_fetched += len(new_posts)
                
                for post in new_posts:
                    post_date = datetime.fromtimestamp(post.created_utc)
                    
                    # Break if we've gone past our cutoff date
                    if cutoff_date and post_date < cutoff_date:
                        pbar.close()
                        logger.info(f"Reached date cutoff in r/{subreddit.display_name}")
                        return
                    
                    posts_yielded += 1
                    yield post
                    pbar.update(1)
                
                # Update pagination parameters
                last_post = new_posts[-1]
                params['after'] = last_post.fullname
                params['count'] = total_fetched
                
                # Avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error fetching batch from r/{subreddit.display_name}: {e}")
                time.sleep(2)
                continue
                
        pbar.close()
        logger.info(f"Completed fetching from r/{subreddit.display_name}: Posts yielded: {posts_yielded:,}")
        
    except Exception as e:
        logger.error(f"Error during pagination: {e}")
        raise

def process_comments(submission: praw.models.Submission, term_patterns: re.Pattern, config: dict) -> Tuple[List[Dict], bool]:
    """Process comments from a submission, excluding AutoModerator comments."""
    
    comments = []
    term_found = False
    comment_limit = config.get('comment_limit', float('inf'))

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
                
            comment_data = {
                'post_id': submission.id,
                'body': comment.body,
                'score': comment.score,
                'contains_term': contains_search_term
            }
            comments.append(comment_data)
            
        return comments, term_found
        
    except Exception as e:
        logger.warning(f"Error processing comments for submission {submission.id}: {e}")
        return [], False

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

def process_submission(submission: praw.models.Submission, search_term: str, config: dict) -> Optional[Tuple[Dict, List[Dict]]]:
    """Process a single submission and its comments."""
    try:
        term_patterns = compile_term_patterns(search_term)
        
        # Check title and body for search term
        title_contains = contains_term(submission.title, term_patterns)
        body_contains = contains_term(submission.selftext, term_patterns)
        
        # Process comments and check for search term
        comments, comments_contain = process_comments(submission, term_patterns, config)
        
        # If term isn't found anywhere, skip this submission
        if not (title_contains or body_contains or comments_contain):
            return None
            
        media_info = check_media_type(submission)
        
        post_data = {
            'post_id': submission.id,
            'title': submission.title,
            'selftext': submission.selftext,
            'score': submission.score,
            'subreddit': submission.subreddit.display_name,
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

def process_subreddit(subreddit_name: str, reddit: praw.Reddit, config: dict) -> Tuple[List[dict], List[dict]]:
    """Process a single subreddit and return collected posts and comments."""
    subreddit_posts = []
    subreddit_comments = []
    search_term = config['search_term']
    post_limit = config.get('post_limit', float('inf'))
    
    try:
        logger.info(f"Processing r/{subreddit_name}")
        subreddit = reddit.subreddit(subreddit_name)
        post_count = 0
        
        for submission in get_top_posts(subreddit, config):
            try:
                # Add rate limiting delay
                time.sleep(0.5) 
                
                result = process_submission(submission, search_term, config)
                if result is not None:
                    post_data, comments = result
                    subreddit_posts.append(post_data)
                    subreddit_comments.extend(comments)
                    post_count += 1
                    
                    if post_count >= post_limit:
                        logger.info(f"Reached post limit ({post_limit}) for r/{subreddit_name}")
                        break
                        
            except Exception as e:
                logger.warning(f"Error processing submission in r/{subreddit_name}: {e}")
                continue
                
        logger.info(f"Collected {post_count} relevant posts from r/{subreddit_name}")
        
    except Exception as e:
        logger.error(f"Error processing subreddit r/{subreddit_name}: {e}")
        return [], []
        
    return subreddit_posts, subreddit_comments

def save_to_csv(posts: List[Dict], comments: List[Dict], output_folder: str, search_term: str) -> None:
    """Save a batch of posts and comments to CSV files with thread safety."""
    if not posts:
        return
        
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with csv_lock:
        os.makedirs(output_folder, exist_ok=True)
        
        # Save posts
        posts_filename = f"{output_folder}/reddit_{search_term}_posts_{timestamp}.csv"
        posts_df = pd.DataFrame(posts)
        posts_df.to_csv(posts_filename, index=False)
        logger.info(f"Saved {len(posts):,} posts to {posts_filename}")
        
        # Save comments
        if comments:
            comments_filename = f"{output_folder}/reddit_{search_term}_comments_{timestamp}.csv"
            comments_df = pd.DataFrame(comments)
            comments_df.to_csv(comments_filename, index=False)
            logger.info(f"Saved {len(comments):,} comments to {comments_filename}")


def collect_data(config: dict, reddit: praw.Reddit) -> None:
    """Main function to collect Reddit data based on configuration with parallel processing."""
    all_posts = []
    all_comments = []
    start_time = datetime.now()

    max_workers = min(len(config['subreddits']), 5)  # Limit to 5 concurrent workers
    
    # Create thread pool for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all subreddit processing tasks
        future_to_subreddit = {
            executor.submit(process_subreddit, subreddit, reddit, config): subreddit
            for subreddit in config['subreddits']
        }
        
        # Process completed tasks as they finish
        for future in as_completed(future_to_subreddit):
            subreddit = future_to_subreddit[future]
            try:
                subreddit_posts, subreddit_comments = future.result()
                all_posts.extend(subreddit_posts)
                all_comments.extend(subreddit_comments)
            except Exception as e:
                logger.error(f"Error processing r/{subreddit}: {e}")

    # Save results and log statistics
    save_to_csv(all_posts, all_comments, config['output_folder'], config['search_term'])

    elapsed = datetime.now() - start_time
    logger.info("\nCollection completed!")
    logger.info(f"Total posts processed: {len(all_posts):,}")
    logger.info(f"Total comments collected: {len(all_comments):,}")
    logger.info(f"Time elapsed: {elapsed}")

def main():
    """Entry point for the Reddit data collection script."""
    try:
        config = load_config()
        
        global logger
        logger = setup_logging(config)
        reddit = initialize_reddit()
        collect_data(config, reddit)
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise

if __name__ == "__main__":
    main()