import asyncio
import os
import sys
import flask

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Now use an absolute import
from video_processor_v2 import process_video

async def main():
    # Replace this with the path to your test TikTok video file
    video_file = '/home/ec2-user/tiktok-extractor-v2/files/video/7402402958266076432.mp4'
    
    if not os.path.exists(video_file):
        print(f"Error: The video file {video_file} does not exist.")
        return

    print(f"Processing video: {video_file}")
    result = await process_video(video_file)
    
    if result:
        print("Extracted texts:")
        if result['texts']:
            for frame, text in result['texts']:
                print(f"Frame {frame}: {text}")
        else:
            print("No text extracted from the video.")
        
        print("\nAudio transcription:")
        print(result['audio_transcription'])
    else:
        print("Video processing failed.")

if __name__ == "__main__":
    asyncio.run(main())