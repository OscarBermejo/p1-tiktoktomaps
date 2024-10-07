import logging
from celery import Celery
import redis
import logger_config
import sys
import os

# Add the directory above the current one to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logger_config.get_logger(__name__)

# Create Redis client
redis_client = redis.Redis.from_url(Config.CELERY_BROKER_URL)

# Create Celery app
celery_app = Celery('tasks',
                    broker=Config.CELERY_BROKER_URL,
                    backend=Config.CELERY_RESULT_BACKEND)

# Configure Celery
celery_app.conf.update(Config.CELERY_CONFIG)

# Test the Redis connection
try:
    redis_client = redis.Redis.from_url(Config.CELERY_BROKER_URL)
    redis_client.ping()
    logger.info("Successfully connected to Redis")
except Exception as e:
    logger.error(f"Failed to connect to Redis: {str(e)}")

# Print Celery configuration for debugging
print("Celery Configuration:")
print(f"Broker URL: {celery_app.conf.broker_url}")
print(f"Result Backend: {celery_app.conf.result_backend}")

# Include tasks
celery_app.autodiscover_tasks(['tasks'])

# This ensures that the Celery logger is set up correctly
logger_config.get_logger('celery')

logger.info("Celery app configured successfully")

if __name__ == '__main__':
    celery_app.start()

# Test the Redis connection
try:
    redis_client = redis.Redis.from_url(Config.CELERY_BROKER_URL)
    redis_client.ping()
    logger.info("Successfully connected to Redis")
except Exception as e:
    logger.error(f"Failed to connect to Redis: {str(e)}")

if __name__ == '__main__':
    celery_app.start()