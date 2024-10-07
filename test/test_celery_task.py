import os
import sys
from celery import Celery


# Ensure the project root is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
from celery_tasks import process_video
sys.path.insert(0, project_root)

# Create a Celery app for testing
celery_app = Celery('test_app')
celery_app.config_from_object('config.Config')

def test_process_video():
    # Replace this with a valid TikTok URL for testing
    test_url = "https://www.tiktok.com/@rome.travelers/video/7185551271389072682?q=best%20restaurnat%20in%20rome&t=1727801016442"
    
    print(f"Starting test with URL: {test_url}")

    # Call the Celery task
    result = process_video.delay(test_url)
    
    print(f"Task ID: {result.id}")
    print("Waiting for task to complete...")
    
    # Add more debugging information
    print(f"Task state: {result.state}")
    
    try:
        # Wait for the task to complete and get the result
        task_result = result.get(timeout=300)  # 5 minutes timeout
        print("Task completed!")
        print("Result:")
        print(task_result)
    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"Final task state: {result.state}")

if __name__ == "__main__":
    test_process_video()