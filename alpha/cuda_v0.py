import torch
import torchvision.transforms as transforms
from PIL import Image
import subprocess
import os
import logging
import easyocr
import time
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check CUDA availability
use_cuda = torch.cuda.is_available()
logger.info(f"CUDA available: {use_cuda}")

# Make reader a global variable that's initialized lazily
reader = None

def get_reader():
    """Lazy initialization of EasyOCR reader"""
    global reader
    if reader is None:
        logger.info("Initializing EasyOCR reader...")
        model_load_start = time.time()
        
        reader = easyocr.Reader(
            ['en'], 
            gpu=use_cuda, 
            model_storage_directory='./models',
            download_enabled=False,
            quantize=True,  # Enable quantization
            cudnn_benchmark=True,  # Speed up CUDA initialization
            verbose=False
        )
        model_load_time = time.time() - model_load_start
        logger.info(f"EasyOCR reader initialization complete! Time taken: {model_load_time:.2f} seconds")
    return reader

def cleanup_frames():
    """Remove the frames directory and its contents"""
    frames_dir = "frames"
    if os.path.exists(frames_dir):
        logger.info("Cleaning up frames directory...")
        shutil.rmtree(frames_dir)
        logger.info("Frames directory removed")

def extract_frames(video_path):
    """Extract frames using ffmpeg"""
    try:
        # Clean up existing frames directory first
        cleanup_frames()
        
        frames_dir = "frames"
        os.makedirs(frames_dir, exist_ok=True)
        
        # Get video duration using ffprobe
        duration_cmd = [
            'ffprobe', 
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        duration = float(subprocess.check_output(duration_cmd).decode('utf-8').strip())
        
        frame_extract_start = time.time()
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vf', 'fps=0.5',  # Extract one frame every 2 seconds
            '-frame_pts', '1',
            f'{frames_dir}/frame_%d.jpg'  # Will be renamed later
        ]
        
        logger.info("Extracting frames with ffmpeg...")
        subprocess.run(cmd, check=True)
        
        # Rename frames to match the processing order
        for i, filename in enumerate(sorted(os.listdir(frames_dir)), 1):
            old_path = os.path.join(frames_dir, filename)
            new_path = os.path.join(frames_dir, f'processed_frame_{i}.jpg')
            os.rename(old_path, new_path)
            
        frame_extract_time = time.time() - frame_extract_start
        logger.info(f"Frame extraction complete! Time taken: {frame_extract_time:.2f} seconds")
        return True, duration, frame_extract_time
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error extracting frames: {str(e)}")
        return False, 0, 0

def process_video(video_path):
    """Process video and return all text as a single string and timing information"""
    timing_info = {
        'video_duration': 0,
        'frame_extract_time': 0,
        'model_load_time': 0,
        'ocr_process_time': 0,
        'total_time': 0,
        'frame_count': 0
    }
    
    try:
        total_start_time = time.time()
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Extract frames
        frames_success, video_duration, frame_extract_time = extract_frames(video_path)
        if not frames_success:
            raise Exception("Frame extraction failed")
        
        timing_info['video_duration'] = video_duration
        timing_info['frame_extract_time'] = frame_extract_time
        
        # Initialize reader only when needed
        model_load_start = time.time()
        ocr_reader = get_reader()
        model_load_time = time.time() - model_load_start
        timing_info['model_load_time'] = model_load_time
        
        # Process frames and extract text
        frames_dir = "frames"
        all_text = []
        
        ocr_process_start = time.time()
        frame_count = 0
        
        # Process frames in order
        for frame_file in sorted(os.listdir(frames_dir)):
            if not frame_file.endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            frame_path = os.path.join(frames_dir, frame_file)
            frame_count += 1
            
            # Extract text from frame
            frame_start = time.time()
            results = ocr_reader.readtext(frame_path)
            frame_time = time.time() - frame_start
            logger.info(f"Frame {frame_count} processed in {frame_time:.2f} seconds")
            
            # Collect text if found in this frame
            for (_, text, confidence) in results:
                if confidence > 0.5:  # Only include text with confidence > 50%
                    all_text.append(text)
        
        timing_info['ocr_process_time'] = time.time() - ocr_process_start
        timing_info['total_time'] = time.time() - total_start_time
        timing_info['frame_count'] = frame_count
        
        # Join all text with spaces
        return ' '.join(all_text), timing_info
        
    except Exception as e:
        logger.error(f"Error in video processing: {str(e)}")
        return "", timing_info
    finally:
        if use_cuda:
            torch.cuda.empty_cache()
        #cleanup_frames()  # Clean up frames after processing

if __name__ == "__main__":
    script_start_time = time.time()
    video_path = '/home/ec2-user/tiktok-extractor-v2/files/video/7402402958266076432.mp4'
    text, timing_info = process_video(video_path)
    script_total_time = time.time() - script_start_time
    
    print("\n=== Text Found in Video ===")
    print(text)
    print("\n=== Processing Summary ===")
    print(f"Video duration: {timing_info['video_duration']:.2f} seconds")
    print(f"Frame extraction time: {timing_info['frame_extract_time']:.2f} seconds")
    print(f"Model loading time: {timing_info['model_load_time']:.2f} seconds")
    print(f"OCR processing time: {timing_info['ocr_process_time']:.2f} seconds")
    print(f"Total time: {script_total_time:.2f} seconds")
    print(f"Number of frames processed: {timing_info['frame_count']}")
