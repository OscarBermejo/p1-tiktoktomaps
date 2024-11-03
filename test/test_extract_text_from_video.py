import os
import time
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extract_text_from_video_v2 import main as extract_text
import logger_config

logger = logger_config.get_logger(__name__)

def test_video_processing():
    # Test videos directory
    test_dir = "test_videos"
    
    # Create test directory if it doesn't exist
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    # Test cases - you can add more video paths
    test_cases = [
        {
            "video_path": "/home/ec2-user/tiktok-extractor-v2/files/video/7402402958266076432.mp4",
            "video_id": "7402402958266076432"
        },
        # Add more test cases as needed
        # {
        #     "video_path": "path/to/another/video.mp4",
        #     "video_id": "another_video_id"
        # },
    ]
    
    # Run tests
    for test_case in test_cases:
        video_path = test_case["video_path"]
        video_id = test_case["video_id"]
        
        logger.info(f"\n=== Testing video: {video_id} ===")
        logger.info(f"Video path: {video_path}")
        
        try:
            # Check if video file exists
            if not os.path.exists(video_path):
                logger.error(f"Video file not found: {video_path}")
                continue
                
            # Process video and measure time
            start_time = time.time()
            texts, video_duration = extract_text(video_path, video_id)
            total_time = time.time() - start_time
            
            # Print results
            logger.info("\n=== Results ===")
            logger.info(f"Video duration: {video_duration:.2f} seconds")
            logger.info(f"Total processing time: {total_time:.2f} seconds")
            logger.info(f"Processing speed ratio: {total_time/video_duration:.2f}x real-time")
            logger.info(f"Extracted text: {texts}")
            
        except Exception as e:
            logger.error(f"Error processing video {video_id}: {str(e)}")
            logger.exception("Full traceback:")

if __name__ == "__main__":
    logger.info("Starting text extraction tests...")
    test_video_processing()
    logger.info("Tests completed!")