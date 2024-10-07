from celery import Celery
from celery import shared_task
from celery_app import celery_app
from video_processor import analyze_video_content
from data_extractor import extract_data
from models import ProcessedVideo
from utils import search_location, query_chatgpt
import asyncio
import logger_config


logger = logger_config.get_logger(__name__)

@celery_app.task
def process_video(url):
    logger.info(f"Task running on worker")
    logger.info(f"Starting to process video: {url}")

    try:
        logger.info("Extracting data from URL")

        # Run the asynchronous functions using asyncio.run
        video_id, video_file, audio_file, description = asyncio.run(extract_data(url))
        texts, audio_transcription = asyncio.run(analyze_video_content(video_file))

        logger.info("Video content analyzed")
        
        logger.info(f"Description: {description}")
        logger.info(f"audio_transcription: {audio_transcription}")
        logger.info(f"texts: {texts}")
        
        chatgpt_query = f"""
        Description: {description}
        Transcription: {audio_transcription}
        Text in images: {texts}
        
        Given the above information from a TikTok video, can you return any place of interest that is being recommended? 
        Please return the results with one recommendation per row, and if you know the city or town, add it in the same row as comma-separated.
        If you know the type of place that is being recommended (restaurant, town, museum...) also add as comma-separated.
        
        Example: Maseria Moroseta, Italy, Restaurant

        If you cannot find any place of interest, make sure you return as your whole message "No places of interest found".

        """
        
        logger.info("Querying ChatGPT")
        recommendations = query_chatgpt(chatgpt_query)
        if "No places of interest found" in recommendations:
            recommendations = "No places of interest found"
        
            logger.info(f"ChatGPT recommendations: {recommendations}")
        
        google_map_links = {}
        # Split the string into lines
        places = recommendations.splitlines()
        logger.info('Places recommended: ' + str(places))

        # Get Google Maps links
        google_map_dict = {}
        google_map_links = []
        for location in places:
            try:
                location = location.strip()  # Remove any leading/trailing
                location_info = search_location(location)
            except Exception as e:
                logger.error(f"Error searching location: {str(e)}", exc_info=True)
                location_info = ''
                continue

            if location_info:
                
                google_map_links.append(location_info['google_maps_link'])
                google_map_dict[location] = location_info['google_maps_link']
        
        logger.info("Updating database with results")
        ProcessedVideo.update_results(url, google_map_dict)
        
        logger.info("Processing completed successfully")
        return google_map_dict
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        raise