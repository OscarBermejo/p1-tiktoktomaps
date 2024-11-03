import cv2
import boto3
import os
import numpy as np
import time
import pytesseract
import re
from concurrent.futures import ThreadPoolExecutor
from botocore.exceptions import NoCredentialsError
import imagehash
from PIL import Image
from difflib import SequenceMatcher
import sys
import shutil

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logger_config

logger = logger_config.get_logger(__name__)

# AWS configuration
aws_region = 'eu-central-1'  # Change to your AWS region
s3_bucket_name = 'p1-tiktoktomaps'  # Replace with your S3 bucket name

# Initialize AWS clients
s3_client = boto3.client('s3', region_name=aws_region)
textract_client = boto3.client('textract', region_name=aws_region)

# Function to upload a file to S3
def upload_to_s3(file_path, bucket_name, object_name):
    try:
        s3_client.upload_file(file_path, bucket_name, object_name)
    except NoCredentialsError:
        logger.info("AWS credentials not available.")

# Function to extract text from an image using Textract
def extract_text_from_image(s3_bucket, s3_object):
    try:
        response = textract_client.detect_document_text(
            Document={'S3Object': {'Bucket': s3_bucket, 'Name': s3_object}}
        )
        text = ''
        for item in response['Blocks']:
            if item['BlockType'] == 'LINE':
                text += item['Text'] + '\n'
        return text
    except Exception as e:
        logger.info(f"Error in Textract for {s3_object}: {e}")
        return ''
    
def increase_contrast(image):
    alpha = 1.5  # Contrast control
    beta = 0     # Brightness control
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

def reduce_noise(image):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    return blurred

def morphological_processing(image):
    kernel = np.ones((2, 2), np.uint8)
    eroded = cv2.erode(image, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    return dilated

def clean_text(text):
    # Keep only alphanumeric characters, periods, slashes, and spaces
    cleaned = re.sub(r'[^A-Za-z0-9\s./]', '', text)
    return cleaned.strip()

# Function to preprocess image for OCR
def preprocess_image(image):
    # Increase contrast
    contrasted = increase_contrast(image)

    # Reduce noise
    denoised = reduce_noise(contrasted)

    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological processing
    processed = morphological_processing(thresh)

    return processed

# Function to check if a frame contains text using Tesseract
def contains_text(image, lang='eng', min_conf=60, min_text_height=5, min_text_width=5, min_text_density=0.001):
    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Customize Tesseract configuration
    custom_config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/.- --psm 6'

    data = pytesseract.image_to_data(
        preprocessed_image,
        lang=lang,
        config=custom_config,
        output_type=pytesseract.Output.DICT
    )

    text_content = ''
    total_text_area = 0
    image_area = image.shape[0] * image.shape[1]

    for i in range(len(data['text'])):
        conf = int(data['conf'][i])
        text = data['text'][i]
        width = int(data['width'][i])
        height = int(data['height'][i])

        # Check confidence and size
        if conf != -1 and conf >= min_conf and width >= min_text_width and height >= min_text_height:
            text_content += text + ' '
            total_text_area += width * height

    # Calculate text density
    text_density = total_text_area / image_area
    logger.info(f"Text density: {text_density}")

    # Clean the text
    text_content = clean_text(text_content)

    # Return text only if density is above threshold
    return text_content.strip() if text_density >= min_text_density else ''

# Function to process a single frame (upload and extract text)
def process_frame(frame_path, s3_bucket_name, video_id, frame_number):
    s3_object_name = f"frames/{video_id}/frame{frame_number}.jpg"
    upload_to_s3(frame_path, s3_bucket_name, s3_object_name)
    text = extract_text_from_image(s3_bucket_name, s3_object_name)
    return text

# Function to resize an image while keeping the aspect ratio
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

# Function to extract and process frames from a video
def extract_and_process_frames(video_path, output_folder, frame_interval, video_id, max_width=None, max_height=None, lang='eng'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    current_frame = 0

    # Set to store processed frame hashes
    processed_frame_hashes = set()

    # Set to store processed text boxes
    processed_text_boxes = set()

    # List to collect all extracted texts
    all_extracted_texts = []

    with ThreadPoolExecutor(max_workers=3) as executor:
        while True:
            ret = cap.grab()
            if not ret:
                break

            if current_frame % frame_interval == 0:
                ret, frame = cap.retrieve()
                if not ret:
                    break

                # Resize the frame while keeping aspect ratio
                resized_image = resize_image_aspect_ratio(frame, max_width=max_width, max_height=max_height)

                # Convert to grayscale
                gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

                frame_filename = os.path.join(output_folder, f"frame{current_frame}.jpg")

                # Calculate the hash of the current frame
                frame_hash = imagehash.average_hash(Image.fromarray(gray_image))

                # Check if a similar frame has already been processed
                if any(frame_hash - processed_hash <= 10 for processed_hash in processed_frame_hashes):
                    logger.info(f"Frame {current_frame} skipped (similar to a processed frame).")
                    current_frame += 1
                    continue

                cv2.imwrite(frame_filename, gray_image)

                # Check for text using Tesseract
                text_content = contains_text(gray_image, lang=lang)
                if text_content:
                    # Check if a similar text box has already been processed
                    if any(SequenceMatcher(None, text_content, processed_text).ratio() >= 0.6 for processed_text in processed_text_boxes):
                        logger.info(f"Text box in frame {current_frame} skipped (similar to a processed text box).")
                    else:
                        logger.info(f"Extracted {frame_filename}")
                        s3_object_name = f"frames/frame{current_frame}.jpg"
                        # Process the frame in parallel
                        future = executor.submit(process_frame, frame_filename, s3_bucket_name, video_id, current_frame)
                        processed_text_boxes.add(text_content)
                        all_extracted_texts.append(future.result())
                else:
                    logger.info(f"No text detected in frame {current_frame}.")
                    # Optionally remove the frame file if no text is detected
                    # os.remove(frame_filename)

                # Add the frame hash to the set of processed frame hashes
                processed_frame_hashes.add(frame_hash)

            current_frame += 1

    cap.release()
    return all_extracted_texts

# Main function
def main(video_path, video_id, lang='eng'):

    # Remove frames folder if exists
    if os.path.exists('frames'):
        shutil.rmtree('frames')

    # Get video FPS and duration
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = frame_count / fps
    cap.release()

    # Set frame_interval based on desired time between frames (e.g., every 1 second)
    time_between_frames = 2  # seconds
    frame_interval = max(int(fps * time_between_frames), 1)

    # Measure processing time
    start_time = time.time()

    output_folder = 'frames'
    # Resize images to a max width or height while keeping aspect ratio
    max_width = 640  # Adjust as needed
    max_height = None  # Set to None to only use max_width
    all_texts = extract_and_process_frames(video_path, output_folder, frame_interval, video_id, max_width=max_width, max_height=max_height, lang=lang)

    end_time = time.time()
    processing_time = end_time - start_time

    # Print processing time and video duration
    logger.info(f"Processing time: {processing_time:.2f} seconds")
    logger.info(f"Video duration: {video_duration:.2f} seconds")

    print(all_texts)
    print(f"Video duration: {video_duration:.2f} seconds")
    print(f"Processing time: {processing_time:.2f} seconds")

    return all_texts, video_duration

# Run the script
if __name__ == "__main__":
    video_file_path = '/home/ec2-user/tiktok-extractor-v2/files/video/7185551271389072682.mp4'  # Replace with your video file path
    video_id = '7185551271389072682'  # Extract this from the file name or pass it as an argument
    lang = 'eng'
    extracted_texts = main(video_file_path, video_id, lang=lang)
