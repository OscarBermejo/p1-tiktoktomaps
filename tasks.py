from celery import Celery
from celery import shared_task
from celery_app import celery_app
import time
import asyncio
from models import ProcessedVideo

# Import the functions used in run_app.py
from data_extractor import extract_data
from extract_text_from_video import main as extract_text_main
from extract_transcription_from_audio import main as extract_transcription_main
from utils import search_location_v2, query_chatgpt_v2
import logger_config

logger = logger_config.get_logger(__name__)

@celery_app.task
def process_video(url):
    logger.info(f"Task running on worker")
    logger.info(f"Starting to process video: {url}")

    try:
        # Measure processing time
        start_time = time.time()

        # Define an asynchronous inner function to run async code
        async def process():
            # Extract data
            video_id, video_file, audio_file, description = await extract_data(url)
            logger.info(f"Video ID: {video_id}")
            logger.info(f"Video File: {video_file}")
            logger.info(f"Audio File: {audio_file}")
            logger.info(f"Description: {description}")

            # Extract texts from video
            logger.info("Extracting texts from video...")
            texts, video_duration = extract_text_main(video_file)

            # Extract transcription from audio
            logger.info("Extracting transcription from audio...")
            transcription = extract_transcription_main(audio_file)
            logger.info(f"Transcription: {transcription}")

            # Query ChatGPT
            logger.info("Querying ChatGPT...")
            recommendations = query_chatgpt_v2(description, texts, transcription)

            # Search locations on Google Maps
            logger.info("Searching locations on Google Maps...")
            google_map_dict = search_location_v2(recommendations)
            logger.info(f"Google Map Links: {google_map_dict}")

            # Update database with results
            logger.info("Updating database with results")
            ProcessedVideo.update_results(url, google_map_dict)

            return google_map_dict, video_duration

        # Run the async function
        loop = asyncio.get_event_loop()
        google_map_dict, video_duration = loop.run_until_complete(process())

        end_time = time.time()
        processing_time = end_time - start_time
        logger.info(f"Total Processing time: {processing_time:.2f} seconds")
        logger.info(f"Video duration: {video_duration:.2f} seconds")
        logger.info("Processing completed successfully")

        return google_map_dict

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        return {'error': str(e)}  