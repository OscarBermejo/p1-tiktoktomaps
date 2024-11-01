import torch
import torchvision.transforms as transforms
from PIL import Image
import subprocess
import os
import logging
import easyocr

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check CUDA availability
use_cuda = torch.cuda.is_available()
logger.info(f"CUDA available: {use_cuda}")

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=use_cuda)

def extract_frames(video_path):
    """Extract frames using ffmpeg"""
    try:
        frames_dir = "frames"
        os.makedirs(frames_dir, exist_ok=True)
        
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vf', 'fps=0.5',  # Extract one frame every 5 seconds
            '-frame_pts', '1',
            f'{frames_dir}/frame_%d.jpg'
        ]
        
        logger.info("Extracting frames with ffmpeg...")
        subprocess.run(cmd, check=True)
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error extracting frames: {str(e)}")
        return False

def process_video(video_path):
    """Process video and return all text as a single string"""
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Extract frames
        if not extract_frames(video_path):
            raise Exception("Frame extraction failed")
        
        # Process frames and extract text
        frames_dir = "frames"
        all_text = []
        
        for frame_file in sorted(os.listdir(frames_dir)):
            if not frame_file.endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            frame_path = os.path.join(frames_dir, frame_file)
            
            # Extract text from frame
            results = reader.readtext(frame_path)
            
            # Collect text if found in this frame
            for (_, text, confidence) in results:
                if confidence > 0.5:  # Only include text with confidence > 50%
                    all_text.append(text)
        
        # Join all text with spaces
        return ' '.join(all_text)
        
    except Exception as e:
        logger.error(f"Error in video processing: {str(e)}")
        return ""
    finally:
        if use_cuda:
            torch.cuda.empty_cache()

if __name__ == "__main__":
    video_path = '/home/ec2-user/tiktok-extractor-v2/files/video/7402402958266076432.mp4'
    text = process_video(video_path)
    print("\n=== Text Found in Video ===\n")
    print(text)
