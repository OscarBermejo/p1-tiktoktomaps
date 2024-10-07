import logging
import watchtower
import boto3
from datetime import datetime
from celery.signals import after_setup_logger

# Set up log group and log stream base
LOG_GROUP = 'p1-flask-app-logs'
STREAM_BASE = 'log-stream-'

def get_log_stream_name():
    current_date = datetime.now().strftime('%Y-%m-%d')
    return f"{STREAM_BASE}{current_date}"

def setup_logger(logger_name=None):
    log_stream = get_log_stream_name()
    client = boto3.client('logs', region_name='eu-central-1')

    try:
        client.create_log_stream(logGroupName=LOG_GROUP, logStreamName=log_stream)
        print(f"Log stream '{log_stream}' created successfully in log group '{LOG_GROUP}'.")
    except client.exceptions.ResourceAlreadyExistsException:
        print(f"Log stream '{log_stream}' already exists.")
    except Exception as e:
        print(f"ERROR: Failed to create log stream: {e}")
        return None

    cloudwatch_handler = watchtower.CloudWatchLogHandler(
        log_group=LOG_GROUP,
        stream_name=log_stream,
        boto3_client=client,
        create_log_group=False
    )

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    cloudwatch_handler.setFormatter(formatter)

    # Create a stream handler for console output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Get the logger
    if logger_name:
        logger = logging.getLogger(logger_name)
    else:
        logger = logging.getLogger()
    
    logger.setLevel(logging.INFO)

    # Remove any existing handlers to avoid duplicates
    logger.handlers = []

    # Add the handlers
    logger.addHandler(cloudwatch_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized for {logger_name if logger_name else 'root'}.")
    return logger

def get_logger(name):
    return setup_logger(name)

# Celery signal to setup logging for workers
@after_setup_logger.connect
def setup_celery_logger(logger, *args, **kwargs):
    setup_logger('celery')