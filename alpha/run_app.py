import asyncio
import sys
import os
import time

# Add the alpha directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'alpha'))

# Import the function from download_video.py
from download_video import test_extract_data_integration
from extract_text_from_video import main as extract_text_main
from extract_transcription_from_audio import main as extract_transcription_main
from utils import search_location_v2, query_chatgpt_v2
import logger_config

logger = logger_config.get_logger(__name__)


async def main():

    # Measure processing time
    start_time = time.time()

    # Call the asynchronous function
    video_id, video_file, audio_file, description = await test_extract_data_integration()
    logger.info(f"Video ID: {video_id}")
    logger.info(f"Video File: {video_file}")
    logger.info(f"Audio File: {audio_file}")
    logger.info(f"Description: {description}")

    # Call the main function from extract_text_from_video
    logger.info(f"Extracting texts from video...")
    texts, video_duration = extract_text_main(video_file)
    
    # Call the main function from extract_transcription_from_audio
    logger.info(f"Extracting transcription from audio...")
    transcription = extract_transcription_main(audio_file)
    
    # Query ChatGPT
    logger.info(f"Querying ChatGPT...")
    recommendations = query_chatgpt_v2(description, texts, transcription)
    
    # Serching locations in Google Maps
    google_map_dict = search_location_v2(recommendations)
    logger.info(f"Google Map Links: {google_map_dict}")

    end_time = time.time()
    processing_time = end_time - start_time
    logger.info(f"Total Processing time: {processing_time:.2f} seconds")
    logger.info(f"Video duration: {video_duration:.2f} seconds")
if __name__ == '__main__':
    # Run the main function
    asyncio.run(main())