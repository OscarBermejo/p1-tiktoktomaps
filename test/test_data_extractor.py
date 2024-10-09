from flask import Flask, request, jsonify
import asyncio
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Now use an absolute import
from data_extractor import extract_data

async def extract_data_integration():
    url = "https://www.tiktok.com/@lilikaramalikis/video/7402402958266076432?_r=1&_t=8p0lP6zCCpE"
    video_file, audio_file, description = await extract_data(url)
    assert video_file.endswith('.mp4')
    assert audio_file.endswith('.wav')
    assert isinstance(description, str)

if __name__ == '__main__':
    result = asyncio.run(extract_data_integration())