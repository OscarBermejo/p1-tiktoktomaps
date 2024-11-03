import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks import process_video
import logger_config

logger = logger_config.get_logger(__name__)

def test_video_processing():
    # Test URLs - you can add more test cases here
    test_cases = [
        {
            'url': 'https://www.tiktok.com/@peachykeentravels/video/7376610540845567248?q=best%20restaurnat%20in%20barcelona&t=1730651713198',
            'email': 'test@example.com'
        },
        # Add more test cases as needed
    ]

    for test_case in test_cases:
        try:
            logger.info(f"\n=== Testing URL: {test_case['url']} ===")
            
            # Process the video
            result = process_video.apply(args=[test_case['url'], test_case['email']]).get()
            
            # Check if there was an error
            if 'error' in result:
                logger.error(f"Processing failed: {result['error']}")
                continue
            
            # Log the results
            logger.info("\nProcessing Results:")
            logger.info(f"Video ID: {result['video_id']}")
            logger.info(f"Video Duration: {result['video_duration']:.2f} seconds")
            logger.info(f"Processing Time: {result['processing_time']:.2f} seconds")
            logger.info("\nGoogle Maps Results:")
            for location, url in result['results'].items():
                logger.info(f"{location}: {url}")

        except Exception as e:
            logger.error(f"Test failed for URL {test_case['url']}: {str(e)}", exc_info=True)

def main():
    try:
        # Run the test
        test_video_processing()
        
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()