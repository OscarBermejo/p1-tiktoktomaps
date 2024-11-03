import torch
import torchvision.transforms as transforms
from PIL import Image
import subprocess
import os
import logging
import easyocr
import time
import shutil
import boto3
import numpy as np
from botocore.exceptions import NoCredentialsError
import sys
import imagehash
from difflib import SequenceMatcher

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logger_config

logger = logger_config.get_logger(__name__)

# AWS configuration
aws_region = 'eu-central-1'
s3_bucket_name = 'p1-tiktoktomaps'

# Initialize AWS clients
s3_client = boto3.client('s3', region_name=aws_region)
textract_client = boto3.client('textract', region_name=aws_region)

# Check CUDA availability
use_cuda = torch.cuda.is_available()
logger.info(f"CUDA available: {use_cuda}")

if use_cuda:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

# Global reader instance
reader = None

def check_models_exist():
    """Check if required model files exist in the models directory"""
    required_models = [
        './models/craft_mlt_25k.pth',
        './models/english_g2.pth'
    ]
    return all(os.path.exists(model_path) for model_path in required_models)

def get_reader():
    """Lazy initialization of EasyOCR reader"""
    global reader
    if reader is None:
        logger.info("Initializing EasyOCR reader...")
        model_load_start = time.time()
        
        # Create models directory if it doesn't exist
        if not os.path.exists('./models'):
            os.makedirs('./models')
        
        # Check if models exist
        models_exist = check_models_exist()
        logger.info(f"Models already downloaded: {models_exist}")
        
        try:
            reader = easyocr.Reader(
                ['en'], 
                gpu=use_cuda, 
                model_storage_directory='./models',
                download_enabled=not models_exist,  # Enable downloads only if models don't exist
                quantize=True,
                cudnn_benchmark=True,
                verbose=False
            )
        except Exception as e:
            logger.error(f"Error initializing EasyOCR reader: {str(e)}")
            return None
        
        model_load_time = time.time() - model_load_start
        logger.info(f"EasyOCR reader initialization complete! Time taken: {model_load_time:.2f} seconds")
    return reader

def upload_to_s3(file_path, bucket_name, object_name):
    """Upload a file to S3"""
    try:
        s3_client.upload_file(file_path, bucket_name, object_name)
        return True
    except NoCredentialsError:
        logger.info("AWS credentials not available.")
        return False

def cleanup_frames():
    """Remove the frames directory and its contents"""
    frames_dir = "frames"
    if os.path.exists(frames_dir):
        logger.info("Cleaning up frames directory...")
        shutil.rmtree(frames_dir)
        logger.info("Frames directory removed")

def extract_frames(video_path, video_id):
    """Extract frames using ffmpeg"""
    try:
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
            '-vf', 'fps=0.5,scale=640:-1,format=gray',
            '-frame_pts', '1',
            f'{frames_dir}/frame_%d.jpg'
        ]
        
        logger.info("Extracting frames with ffmpeg...")
        subprocess.run(cmd, check=True)
        
        # Rename and upload frames
        processed_frame_hashes = set()
        frame_files = []
        
        for i, filename in enumerate(sorted(os.listdir(frames_dir)), 1):
            old_path = os.path.join(frames_dir, filename)
            new_path = os.path.join(frames_dir, f'processed_frame_{i}.jpg')
            
            # Check for duplicate frames using image hash
            frame_hash = imagehash.average_hash(Image.open(old_path))
            if any(frame_hash - processed_hash <= 10 for processed_hash in processed_frame_hashes):
                logger.info(f"Frame {i} skipped (similar to a processed frame).")
                continue
                
            os.rename(old_path, new_path)
            processed_frame_hashes.add(frame_hash)
            frame_files.append(new_path)
            
            # Upload to S3
            s3_object_name = f"frames/{video_id}/frame{i}.jpg"
            upload_to_s3(new_path, s3_bucket_name, s3_object_name)
            
        frame_extract_time = time.time() - frame_extract_start
        logger.info(f"Frame extraction complete! Time taken: {frame_extract_time:.2f} seconds")
        return True, duration, frame_extract_time, frame_files
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error extracting frames: {str(e)}")
        return False, 0, 0, []

def process_video(video_path, video_id):
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
        frames_success, video_duration, frame_extract_time, frame_files = extract_frames(video_path, video_id)
        if not frames_success:
            raise Exception("Frame extraction failed")
        
        timing_info['video_duration'] = video_duration
        timing_info['frame_extract_time'] = frame_extract_time
        
        # Initialize reader
        model_load_start = time.time()
        ocr_reader = get_reader()
        model_load_time = time.time() - model_load_start
        timing_info['model_load_time'] = model_load_time
        
        # Process frames and extract text
        all_text = []
        processed_text_boxes = set()
        
        ocr_process_start = time.time()
        frame_count = 0
        
        # Process frames in order
        for frame_path in frame_files:
            frame_count += 1
            
            # Extract text from frame
            frame_start = time.time()
            results = ocr_reader.readtext(frame_path)
            frame_time = time.time() - frame_start
            logger.info(f"Frame {frame_count} processed in {frame_time:.2f} seconds")
            
            # Process results
            for (_, text, confidence) in results:
                if confidence > 0.5:
                    # Check for similar text already processed
                    if not any(SequenceMatcher(None, text, processed_text).ratio() >= 0.6 
                             for processed_text in processed_text_boxes):
                        all_text.append(text)
                        processed_text_boxes.add(text)
        
        timing_info['ocr_process_time'] = time.time() - ocr_process_start
        timing_info['total_time'] = time.time() - total_start_time
        timing_info['frame_count'] = frame_count
        
        return all_text, timing_info
        
    except Exception as e:
        logger.error(f"Error in video processing: {str(e)}")
        return [], timing_info
    finally:
        if use_cuda:
            torch.cuda.empty_cache()

def main(video_path, video_id):
    start_time = time.time()
    
    # Process video
    all_texts, timing_info = process_video(video_path, video_id)
    
    # Print processing summary
    logger.info("\n=== Processing Summary ===")
    logger.info(f"Video duration: {timing_info['video_duration']:.2f} seconds")
    logger.info(f"Frame extraction time: {timing_info['frame_extract_time']:.2f} seconds")
    logger.info(f"Model loading time: {timing_info['model_load_time']:.2f} seconds")
    logger.info(f"OCR processing time: {timing_info['ocr_process_time']:.2f} seconds")
    logger.info(f"Total time: {timing_info['total_time']:.2f} seconds")
    logger.info(f"Number of frames processed: {timing_info['frame_count']}")
    
    # Join all texts with spaces
    combined_text = ' '.join(all_texts)
    
    return combined_text, timing_info['video_duration']

