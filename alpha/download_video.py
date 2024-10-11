from flask import Flask, request, jsonify
import asyncio
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Now use an absolute import
from data_extractor import extract_data
import logger_config

async def extract_data_integration():
    url = "https://www.tiktok.com/@desarab33/video/7089524798627253510?q=best%20restaurant%20in%20barcelona&t=1728658615290"
    video_id, video_file, audio_file, description = await extract_data(url)
    return video_id, video_file, audio_file, description

if __name__ == '__main__':
    result = asyncio.run(extract_data_integration())