import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now use an absolute import
from app import app, celery_app
from tasks import process_video

import time
from celery.result import AsyncResult
import logger_config

logger = logger_config.get_logger(__name__)

def test_process_video():
    # Test URL - replace with a valid TikTok URL
    test_url = "https://www.tiktok.com/@rome.travelers/video/7185551271389072682?q=best%20restaurnat%20in%20rome&t=1727801016442"
    test_email = "obermejo@live.com"  # Add a test email

    with app.app_context():
        logger.info(f"Starting test with URL: {test_url}")

        # Start the Celery task with both url and email
        task = process_video.delay(test_url, test_email)
        logger.info(f"Task started with ID: {task.id}")

        # Wait for the task to complete
        start_time = time.time()
        while not task.ready():
            task_status = AsyncResult(task.id, app=celery_app)
            logger.info(f"Task is still processing... State: {task_status.state}")
            if time.time() - start_time > 300:  # 5 minutes timeout
                logger.error("Task processing timeout after 5 minutes")
                break
            time.sleep(5)  # Wait for 5 seconds before checking again

        if task.ready():
            # Task is complete, get the result
            try:
                result = task.get(timeout=10)  # 10 seconds timeout for getting result
                logger.info(f"Task completed. Result: {result}")
            except Exception as e:
                logger.error(f"Error getting task result: {str(e)}")
        else:
            logger.error("Task did not complete in time")

        # Check the task status using the check_task route
        with app.test_client() as client:
            response = client.get(f'/check_task/{task.id}')
            logger.info(f"Check task response: {response.json}")

        # Try to get the result using the result route
        with app.test_client() as client:
            response = client.get(f'/result/{task.id}')
            logger.info(f"Result route response status: {response.status_code}")
            logger.info(f"Result route response data: {response.data}")

if __name__ == "__main__":
    test_process_video()