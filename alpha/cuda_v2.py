import cv2
import boto3
import os
import numpy as np
import time
import easyocr
import re
from concurrent.futures import ThreadPoolExecutor
from botocore.exceptions import NoCredentialsError
import imagehash
from PIL import Image
from difflib import SequenceMatcher
import sys
import shutil
from multiprocessing import Process, Queue, freeze_support
import torch
import multiprocessing

# Set multiprocessing start method
multiprocessing.set_start_method('spawn', force=True)

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

# Check if CUDA is available
use_cuda = torch.cuda.is_available()

class OCRServer(Process):
    def __init__(self, input_queue, output_queue):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        
    def run(self):
        try:
            logger.info("Initializing EasyOCR reader in background process...")
            
            # Check if models directory exists
            models_dir = './models'
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
                logger.info("Created models directory. First run will download required models...")
            else:
                logger.info("Found existing models directory, loading local models...")

            reader = easyocr.Reader(
                ['en'], 
                gpu=use_cuda,
                model_storage_directory=models_dir,
                download_enabled=False,  # Set to False after first successful download
                verbose=False,
                cudnn_benchmark=True,
                quantize=True
            )
            logger.info("EasyOCR reader initialization complete!")
            
            while True:
                frame_path = self.input_queue.get()
                if frame_path == "STOP":
                    break
                try:
                    results = reader.readtext(frame_path)
                    self.output_queue.put(results)
                except Exception as e:
                    logger.error(f"Error processing frame {frame_path}: {str(e)}")
                    self.output_queue.put([])
        except Exception as e:
            logger.error(f"Error in OCR server: {str(e)}")
            self.output_queue.put([])

def init_ocr_server():
    input_queue = Queue()
    output_queue = Queue()
    ocr_server = OCRServer(input_queue, output_queue)
    ocr_server.start()
    return input_queue, output_queue, ocr_server

def clean_text(text):
    cleaned = re.sub(r'[^A-Za-z0-9\s./]', '', text)
    return cleaned.strip()

def resize_image_aspect_ratio(image, max_width=None, max_height=None):
    (h, w) = image.shape[:2]
    if max_width is None and max_height is None:
        return image
    if max_width is not None:
        r = max_width / float(w)
        dim = (max_width, int(h * r))
    else:
        r = max_height / float(h)
        dim = (int(w * r), max_height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

def extract_and_process_frames(video_path, output_folder, frame_interval, video_id, input_queue, output_queue, max_width=None, max_height=None):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    current_frame = 0

    processed_frame_hashes = set()
    processed_text_boxes = set()
    all_extracted_texts = []

    while True:
        ret = cap.grab()
        if not ret:
            break

        if current_frame % frame_interval == 0:
            ret, frame = cap.retrieve()
            if not ret:
                break

            resized_image = resize_image_aspect_ratio(frame, max_width=max_width, max_height=max_height)
            gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            frame_filename = os.path.join(output_folder, f"frame{current_frame}.jpg")

            frame_hash = imagehash.average_hash(Image.fromarray(gray_image))

            if any(frame_hash - processed_hash <= 10 for processed_hash in processed_frame_hashes):
                logger.info(f"Frame {current_frame} skipped (similar to a processed frame).")
                current_frame += 1
                continue

            cv2.imwrite(frame_filename, gray_image)

            # Send frame to OCR server
            input_queue.put(frame_filename)
            results = output_queue.get()

            text_content = ""
            for (_, text, confidence) in results:
                if confidence > 0.5:
                    text_content += text + " "

            text_content = clean_text(text_content)

            if text_content:
                if any(SequenceMatcher(None, text_content, processed_text).ratio() >= 0.6 
                      for processed_text in processed_text_boxes):
                    logger.info(f"Text box in frame {current_frame} skipped (similar content).")
                else:
                    logger.info(f"Text extracted from frame {current_frame}")
                    processed_text_boxes.add(text_content)
                    all_extracted_texts.append(text_content)
            else:
                logger.info(f"No text detected in frame {current_frame}.")

            processed_frame_hashes.add(frame_hash)

        current_frame += 1

    cap.release()
    return all_extracted_texts

def process_video_with_ocr(video_path, video_id, input_queue, output_queue):
    """Process a single video and extract text"""
    try:
        # Verify video file exists
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return [], 0, 0

        if os.path.exists('frames'):
            shutil.rmtree('frames')
        os.makedirs('frames', exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            return [], 0, 0

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps == 0 or frame_count == 0:
            logger.error("Invalid video file: Could not get FPS or frame count")
            return [], 0, 0
            
        video_duration = frame_count / fps
        cap.release()

        time_between_frames = 2
        frame_interval = max(int(fps * time_between_frames), 1)

        start_time = time.time()

        output_folder = 'frames'
        max_width = 640
        all_texts = extract_and_process_frames(
            video_path, 
            output_folder, 
            frame_interval, 
            video_id, 
            input_queue,
            output_queue,
            max_width=max_width
        )

        end_time = time.time()
        processing_time = end_time - start_time

        logger.info(f"Processing time: {processing_time:.2f} seconds")
        logger.info(f"Video duration: {video_duration:.2f} seconds")

        return all_texts, video_duration, processing_time

    except Exception as e:
        logger.error(f"Error in video processing: {str(e)}")
        return [], 0, 0
    finally:
        if use_cuda:
            torch.cuda.empty_cache()

def process_video(video_path, video_id):
    # Initialize OCR server
    input_queue, output_queue, ocr_server = init_ocr_server()
    
    try:
        extracted_texts, duration, processing_time = process_video_with_ocr(video_path, video_id, input_queue, output_queue)
        return extracted_texts, duration, processing_time
    finally:
        # Clean up OCR server
        input_queue.put("STOP")
        ocr_server.join()

def main():
    """Main entry point of the script"""
    freeze_support()
    try:
        video_file_path = '/home/ec2-user/tiktok-extractor-v2/files/video/7402402958266076432.mp4'
        video_id = '7402402958266076432'
        
        # Verify video path
        if not os.path.exists(video_file_path):
            logger.error(f"Video file not found: {video_file_path}")
            return
            
        extracted_texts, duration, processing_time = process_video(video_file_path, video_id)
        print(f"Extracted texts: {extracted_texts}")
        
    except Exception as e:
        logger.error(f"Main script error: {str(e)}")

if __name__ == "__main__":
    main()