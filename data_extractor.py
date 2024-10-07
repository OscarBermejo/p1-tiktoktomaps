import yt_dlp
import aiohttp
from bs4 import BeautifulSoup
import json
from functools import lru_cache
import asyncio
import ssl
import certifi
import logger_config
import os
import ffmpeg

logger = logger_config.get_logger(__name__)

@lru_cache(maxsize=100)
def extract_video_id(url):
    logger.info(f"Extracting video ID for URL: {url}")
    ydl_opts = {'quiet': True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            video_id = info['id']
            logger.info(f"Successfully extracted video ID: {video_id}")
            return video_id
    except Exception as e:
        logger.error(f"Error extracting video ID: {str(e)}")
        raise

async def download_tiktok_video(url, video_id):
    logger.info(f"Downloading TikTok video for URL: {url}, Video ID: {video_id}")
    output_file = f'/home/ec2-user/tiktok-extractor-v2/files/video/{video_id}.mp4'
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': output_file,
        'quiet': True
    }
    
    def download():
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            logger.info(f"Successfully downloaded video to: {output_file}")
            return output_file
        except Exception as e:
            logger.error(f"Error downloading video: {str(e)}")
            raise
    
    return await asyncio.to_thread(download)

async def extract_description_from_html(url):
    logger.info(f"Extracting description from HTML for URL: {url}")
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, ssl=ssl_context) as response:
                content = await response.text()
        except Exception as e:
            logger.error(f"Error fetching HTML content: {str(e)}")
            raise
    
    soup = BeautifulSoup(content, 'html.parser')
    for script in soup.find_all('script'):
        if script.string:
            try:
                json_data = json.loads(script.string)
                logger.debug(f"JSON data structure: {json.dumps(json_data, indent=2)[:500]}...")  # Log first 500 chars of the structure
                
                # Ensure json_data is a dictionary
                if isinstance(json_data, dict) and isinstance(json_data.get('__DEFAULT_SCOPE__'), list):
                    for item in json_data['__DEFAULT_SCOPE__']:
                        if isinstance(item, dict) and 'webapp.video-detail' in item:
                            description = item['webapp.video-detail']['itemInfo']['itemStruct']['desc']
                            logger.info(f"Successfully extracted description: {description[:50]}...")  # Log first 50 chars
                            return description
                elif isinstance(json_data.get('__DEFAULT_SCOPE__'), dict):
                    # Original approach
                    description = json_data['__DEFAULT_SCOPE__']['webapp.video-detail']['itemInfo']['itemStruct']['desc']
                    logger.info(f"Successfully extracted description: {description[:50]}...")  # Log first 50 chars
                    return description
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f"Error parsing JSON data: {str(e)}")
                continue
    
    logger.warning("Could not extract description from HTML")
    return ""

async def extract_audio_from_video(video_file):
    logger.info(f"Extracting audio from video file: {video_file}")
    video_id = os.path.splitext(os.path.basename(video_file))[0]
    output_file = f'/home/ec2-user/tiktok-extractor-v2/files/audio/{video_id}.wav'
    
    def extract_audio():
        try:
            (
                ffmpeg
                .input(video_file)
                .output(output_file, acodec='pcm_s16le', ar='44100')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            logger.info(f"Successfully extracted audio to: {output_file}")
            return output_file
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            raise
    
    return await asyncio.to_thread(extract_audio)

async def extract_data(url):
    logger.info(f"Starting data extraction for URL: {url}")
    try:
        video_id = extract_video_id(url)
        video_file, description = await asyncio.gather(
            download_tiktok_video(url, video_id),
            extract_description_from_html(url)
        )
        audio_file = await extract_audio_from_video(video_file)
        logger.info(f"Data extraction completed. Video file: {video_file}, Audio file: {audio_file}, Description length: {len(description)}")
        return video_id, video_file, audio_file, description
    except Exception as e:
        logger.error(f"Error in data extraction: {str(e)}")
        raise
