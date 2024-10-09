import os
import sys
from celery import Celery


# Ensure the project root is in the Python path
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tasks import process_video

# Create a Celery app for testing
celery_app = Celery('test_app')
celery_app.config_from_object('config.Config')

def test_process_video():
    # Replace this with a valid TikTok URL for testing
    test_url = "https://www.tiktok.com/@rome.travelers/video/7185551271389072682?q=best%20restaurnat%20in%20rome&t=1727801016442"
    
    print(f"Starting test with URL: {test_url}")

    # Call the Celery task
    result = process_video.delay(test_url)

    print(f'Result: {result}')
    
    print(f"Task ID: {result.id}")
    print("Waiting for task to complete...")
    
    # Add more debugging information
    print(f"Task state: {result.state}")

    # Wait for the task to complete and get the result
    task_result = result.get(timeout=5)  # 5 minutes timeout
    print("Task completed!")
    print("Result:")
    print(task_result)

if __name__ == "__main__":
    test_process_video()