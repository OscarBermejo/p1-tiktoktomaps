import asyncio
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor

# Add the alpha directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'alpha'))

# Import the function from download_video.py
from download_video import extract_data_integration
from extract_text_from_video import main as extract_text_main
from extract_transcription_from_audio import main as extract_transcription_main
from utils import search_location_v2, query_chatgpt_v2
import logger_config

logger = logger_config.get_logger(__name__)

async def main():
    # Measure processing time
    start_time = time.time()

    # Call the asynchronous function
    video_id, video_file, audio_file, description = await extract_data_integration()
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

if __name__ == '__main__':
    # Run the main function
    asyncio.run(main())