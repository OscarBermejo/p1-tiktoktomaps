from celery import Celery
from celery import shared_task
from celery_app import celery_app
import time
import asyncio
from models import ProcessedVideo
from concurrent.futures import ThreadPoolExecutor

# Import the functions used in run_app.py
from download_video import extract_data
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
            # Call the asynchronous function
            video_id, video_file, audio_file, description = await extract_data(url)
            logger.info(f"Video ID: {video_id}")
            logger.info(f"Video File: {video_file}")
            logger.info(f"Audio File: {audio_file}")
            logger.info(f"Description: {description}")

            # Create a ThreadPoolExecutor with 2 worker threads
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit the extract_text_from_video function to the executor
                text_future = executor.submit(extract_text_main, video_file, video_id)
                
                # Submit the extract_transcription_from_audio function to the executor
                transcription_future = executor.submit(extract_transcription_main, audio_file)
                
                # Wait for both functions to complete and get the results
                texts, video_duration = text_future.result()
                transcription = transcription_future.result()

            # Query ChatGPT
            logger.info(f"Querying ChatGPT...")
            recommendations = query_chatgpt_v2(description, texts, transcription)
            
            # Searching locations in Google Maps
            google_map_dict = search_location_v2(recommendations)
            logger.info(f"Google Map Links: {google_map_dict}")

            end_time = time.time()
            processing_time = end_time - start_time
            logger.info(f"Total Processing time: {processing_time:.2f} seconds")
            logger.info(f"Video duration: {video_duration:.2f} seconds")

            # Update database with results
            logger.info("Updating database with results")
            ProcessedVideo.update_results(url, google_map_dict, video_id, video_duration, processing_time)

            return google_map_dict, video_duration, video_id, processing_time

        # Run the async function
        loop = asyncio.get_event_loop()
        google_map_dict, video_duration, video_id, processing_time = loop.run_until_complete(process())

        logger.info(f"Total Processing time: {processing_time:.2f} seconds")
        logger.info(f"Video duration: {video_duration:.2f} seconds")
        logger.info("Processing completed successfully")

        return {
            'results': google_map_dict, 
            'video_id': video_id, 
            'video_duration': video_duration, 
            'processing_time': processing_time
        }

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        return {'error': str(e)}