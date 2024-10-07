import re
import googlemaps
import logger_config 
import openai
import sys
import os

# Add the directory above the current one to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config

logger = logger_config.get_logger(__name__)

def validate_tiktok_url(url):
    tiktok_pattern = r'^https?:\/\/(www\.)?tiktok\.com\/@[\w.-]+\/video\/\d+(\?.*)?$'
    return re.match(tiktok_pattern, url) is not None

gmaps = googlemaps.Client(key=Config.GOOGLE_MAPS_API_KEY)

def search_location(location_name):
    result = gmaps.places(query=location_name)
    if result['status'] == 'OK':
        place = result['results'][0]
        return {
            'name': place['name'],
            'address': place.get('formatted_address', 'No address found'),
            'latitude': place['geometry']['location']['lat'],
            'longitude': place['geometry']['location']['lng'],
            'google_maps_link': f"https://www.google.com/maps/place/?q=place_id:{place['place_id']}"
        }
    return None

def query_chatgpt(text):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",  
        messages=[{"role": "user", "content": text}],
        max_tokens=150,
        temperature=0.5
    )
    return response.choices[0].message.content

def query_chatgpt_v2(description, text, transcription):

    chatgpt_query = f"""
        Description: {description}
        Transcription: {transcription}
        Text in images: {text}
        
        Given the above information from a TikTok video, can you return any place of interest that is being recommended? 
        Please return the results with one recommendation per row, and if you know the city or town, add it in the same row as comma-separated.
        If you know the type of place that is being recommended (restaurant, town, museum...) also add as comma-separated.
        
        Example: Maseria Moroseta, Italy, Restaurant

        If you cannot find any place of interest, make sure you return as your whole message "No places of interest found".

    """
    response = openai.chat.completions.create(
        model="gpt-4o-mini",  
        messages=[{"role": "user", "content": chatgpt_query}],
        max_tokens=150,
        temperature=0.5
    )
    return response.choices[0].message.content

def search_location_v2(recommendations):

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

            result = gmaps.places(query=location)
            if result['status'] == 'OK':
                place = result['results'][0]
                location_info = {
                    'name': place['name'],
                    'address': place.get('formatted_address', 'No address found'),
                    'latitude': place['geometry']['location']['lat'],
                    'longitude': place['geometry']['location']['lng'],
                    'google_maps_link': f"https://www.google.com/maps/place/?q=place_id:{place['place_id']}"
                    }
        
        except Exception as e:
            logger.error(f"Error searching location: {str(e)}", exc_info=True)
            location_info = ''
            continue

        if location_info:
            
            google_map_links.append(location_info['google_maps_link'])
            google_map_dict[location] = location_info['google_maps_link']

    return google_map_dict