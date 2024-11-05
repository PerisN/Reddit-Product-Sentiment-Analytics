# Import libraries
import os
import praw
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path
import yaml
from urllib.parse import urlparse
import time
from prawcore.exceptions import ResponseException

"""
Reddit Data Collection Script
----------------------------
Purpose:
    This script collects and analyzes top posts from specified subreddits, focusing on gathering
    comprehensive data about post engagement, media content, and user interactions.

Main Features:
    - Collects posts from multiple subreddits over a configurable time period
    - Tracks media content (images, videos, galleries)
    - Gathers and sorts top comments
    - Exports data to CSV format with timestamps
    - Implements rate limiting and error handling
    
Required Environment Variables:
    - CLIENT_ID: Reddit API client ID
    - CLIENT_SECRET: Reddit API client secret
    - USER_AGENT: Reddit API user agent string

Configuration (via config.yaml):
    - subreddits: List of subreddit names to collect from
    - post_limit: Number of top posts to include in final dataset
    - comment_limit: Number of top comments to collect per post
    - months: Number of months of historical data to collect
"""

# Set up logging with timestamp and level information
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_file='config.yaml'):
    """
    Load configuration settings from YAML file.
    Returns dict containing script configuration parameters.
    """
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

# Load configuration
config = load_config()

# Initialize Reddit API connection with error handling
try:
    reddit = praw.Reddit(
        client_id=os.getenv('CLIENT_ID'),
        client_secret=os.getenv('CLIENT_SECRET'),
        user_agent=os.getenv('USER_AGENT')
    )
    reddit.user.me()  # Verify authentication
except Exception as e:
    logging.error(f"Reddit authentication failed: {e}")
    exit(1)

def is_image_url(url):
    """
    Check if a URL points to an image file.
    
    Args:
        url (str): URL to check
        
    Returns:
        bool: True if URL ends with common image extensions
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif']
    parsed_url = urlparse(url)
    return any(parsed_url.path.lower().endswith(ext) for ext in image_extensions)


def get_media_info(submission):
    """Extract media information from submission."""
    media_info = {
        'has_media': False,
        'media_type': None,
        'media_urls': [],
        'item_count': 0,
    }

    try:
        if hasattr(submission, 'is_gallery') and submission.is_gallery:
            media_info['has_media'] = True
            media_info['media_type'] = 'gallery'
            if hasattr(submission, 'media_metadata'):
                media_info['item_count'] = len(submission.media_metadata)
                for item in submission.media_metadata.values():
                    if 's' in item and 'u' in item['s']:
                        media_info['media_urls'].append(item['s']['u'])
            return media_info

        if submission.url and is_image_url(submission.url):
            media_info['has_media'] = True
            media_info['media_type'] = 'image'
            media_info['media_urls'].append(submission.url)
            media_info['item_count'] = 1
            return media_info

        if hasattr(submission, 'media') and submission.media:
            if 'reddit_video' in submission.media:
                media_info['has_media'] = True
                media_info['media_type'] = 'video'
                media_info['media_urls'].append(submission.media['reddit_video']['fallback_url'])
                media_info['item_count'] = 1
            elif 'type' in submission.media:
                media_info['has_media'] = True
                media_info['media_type'] = submission.media['type']
                media_info['item_count'] = 1
                if 'oembed' in submission.media:
                    media_info['media_urls'].append(submission.media['oembed'].get('thumbnail_url', ''))

    except Exception as e:
        logger.error(f"Error processing media for submission {submission.id}: {e}")

    return media_info

def fetch_comments(submission, limit=10):
    """
    Fetch top comments from a submission, excluding AutoModerator.
    
    Args:
        submission: Reddit submission object
        limit (int): Maximum number of comments to fetch
        
    Returns:
        list: List of dicts containing comment data
    """
    comments = []
    submission.comments.replace_more(limit=0)  # Expand comment tree
    sorted_comments = sorted(submission.comments.list(), key=lambda x: x.score, reverse=True)
    
    for comment in sorted_comments:
        if str(comment.author).lower() != 'automoderator':
            comments.append({
                'comment_body': comment.body[:500],  # Truncate long comments
                'comment_score': comment.score,
            })
            if len(comments) >= limit:
                break
    return comments

def fetch_posts_from_subreddits(subreddits, comment_limit, months):
    """
    Fetch posts from multiple subreddits within specified time period.
    
    Args:
        subreddits (list): List of subreddit names
        comment_limit (int): Number of comments to fetch per post
        months (int): Number of months of historical data to collect
        
    Returns:
        list: List of dicts containing post data
    """
    all_posts = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30*months)
    
    for subreddit in subreddits:
        try:
            subreddit_instance = reddit.subreddit(subreddit)
            submissions = subreddit_instance.new(limit=None)
            
            for submission in submissions:
                submission_date = datetime.fromtimestamp(submission.created_utc)
                if submission_date < start_date:
                    break  # Stop if we've gone past our time limit
                if submission_date > end_date:
                    continue
                
                media_info = get_media_info(submission)
                
                # Collect post metadata
                post_data = {
                    'subreddit': subreddit,
                    'post_id': submission.id,
                    'created_utc': submission_date.strftime('%Y-%m-%d %H:%M:%S'),
                    'post_flair': submission.link_flair_text,
                    'title': submission.title,
                    'selftext': submission.selftext[:500],  # Truncate long text
                    'score': submission.score,
                    'upvote_ratio': submission.upvote_ratio,
                    'num_comments': submission.num_comments,
                    'has_image': media_info['has_image'],
                    'has_video': media_info['has_video'],
                    'media_count': media_info['media_count'],
                    'image_url': media_info['image_url'],
                    'video_url': media_info['video_url']
                }
                
                # Add gallery metadata if present
                if media_info['media_metadata']:
                    post_data['gallery_item_count'] = len(media_info['media_metadata'])
                
                # Fetch and add top comments
                comments = fetch_comments(submission, limit=comment_limit)
                for i, comment in enumerate(comments):
                    for key, value in comment.items():
                        post_data[f'{key}_{i+1}'] = value
                        
                all_posts.append(post_data)
                
                # Rate limiting
                time.sleep(0.5)
            
            logging.info(f"Fetched posts from r/{subreddit}")
            
        except ResponseException as e:
            if e.response.status_code == 429:
                logging.warning(f"Rate limit reached for r/{subreddit}. Waiting for 60 seconds before retrying...")
                time.sleep(60)
            else:
                logging.error(f"Error fetching posts from r/{subreddit}: {e}")
        except Exception as e:
            logging.error(f"Error fetching posts from r/{subreddit}: {e}")
    
    return all_posts

def main():
    """
    Main execution function. Coordinates the data collection process and saves results.
    """
    subreddits_list = config['subreddits']
    post_limit = config['post_limit']
    comment_limit = config['comment_limit']
    months = config['months']
    
    logging.info(f"Starting data collection for top {post_limit} posts in the past {months} months from all specified subreddits...")
    all_posts = fetch_posts_from_subreddits(subreddits_list, comment_limit=comment_limit, months=months)
    
    if not all_posts:
        logging.warning("No data collected. Exiting.")
        return
    
    # Sort posts by score and take top posts according to limit
    all_posts.sort(key=lambda x: x['score'], reverse=True)
    top_posts = all_posts[:post_limit]
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(top_posts)
    Path("data").mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/reddit_top_posts_{timestamp}.csv"
    
    try:
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        logging.info(f"Data saved to {filename}")
        logging.info(f"Total posts collected: {len(df)}")
        logging.info(f"Sample of collected data:\n{df.sample(n=5 if len(df) >= 5 else len(df))}")
    except Exception as e:
        logging.error(f"Error saving data to CSV: {e}")

if __name__ == "__main__":
    main()