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
        # print(f"Uploaded {file_path} to s3://{bucket_name}/{object_name}")
    except NoCredentialsError:
        print("AWS credentials not available.")

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
        print(f"Error in Textract for {s3_object}: {e}")
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

    # Invert image if necessary (uncomment if text is white on black background)
    # processed = cv2.bitwise_not(processed)

    return processed

# Function to check if a frame contains text using Tesseract
def contains_text(image, lang='eng', min_conf=50):
    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Save preprocessed image for debugging (optional)
    # cv2.imwrite('preprocessed_frame.jpg', preprocessed_image)

    # Customize Tesseract configuration
    custom_config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/.- --psm 6'

    data = pytesseract.image_to_data(
        preprocessed_image,
        lang=lang,
        config=custom_config,
        output_type=pytesseract.Output.DICT
    )

    text_boxes = len(data['text'])
    text_content = ''
    for i in range(text_boxes):
        conf = data['conf'][i]
        # Check if conf is not -1 (which Tesseract uses to indicate no confidence value)
        if conf != -1 and int(conf) >= min_conf:
            text_content += data['text'][i] + ' '

    # Clean the text
    text_content = clean_text(text_content)

    return text_content.strip()

# Function to process a single frame (upload and extract text)
def process_frame(frame_path, s3_bucket_name, s3_object_name):
    upload_to_s3(frame_path, s3_bucket_name, s3_object_name)
    text = extract_text_from_image(s3_bucket_name, s3_object_name)
    print(f"Text from {frame_path}:\n{text}")

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
def extract_and_process_frames(video_path, output_folder, frame_interval, max_width=None, max_height=None, lang='eng'):
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

    with ThreadPoolExecutor(max_workers=4) as executor:
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
                    print(f"Frame {current_frame} skipped (similar to a processed frame).")
                    current_frame += 1
                    continue

                cv2.imwrite(frame_filename, gray_image)

                # Check for text using Tesseract
                text_content = contains_text(gray_image, lang=lang)
                if text_content:
                    # Check if a similar text box has already been processed
                    if any(SequenceMatcher(None, text_content, processed_text).ratio() >= 0.7 for processed_text in processed_text_boxes):
                        print(f"Text box in frame {current_frame} skipped (similar to a processed text box).")
                    else:
                        print(f"Extracted {frame_filename}")
                        s3_object_name = f"frames/frame{current_frame}.jpg"
                        # Process the frame in parallel
                        executor.submit(process_frame, frame_filename, s3_bucket_name, s3_object_name)
                        processed_text_boxes.add(text_content)
                else:
                    print(f"No text detected in frame {current_frame}.")
                    # Optionally remove the frame file if no text is detected
                    # os.remove(frame_filename)

                # Add the frame hash to the set of processed frame hashes
                processed_frame_hashes.add(frame_hash)

            current_frame += 1

    cap.release()

# Main function
def main(video_path, lang='eng'):
    # Get video FPS and duration
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = frame_count / fps
    cap.release()

    # Set frame_interval based on desired time between frames (e.g., every 1 second)
    time_between_frames = 1  # seconds
    frame_interval = max(int(fps * time_between_frames), 1)

    # Measure processing time
    start_time = time.time()

    output_folder = 'frames'
    # Resize images to a max width or height while keeping aspect ratio
    max_width = 640  # Adjust as needed
    max_height = None  # Set to None to only use max_width
    extract_and_process_frames(video_path, output_folder, frame_interval, max_width=max_width, max_height=max_height, lang=lang)

    end_time = time.time()
    processing_time = end_time - start_time

    # Print processing time and video duration
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Video duration: {video_duration:.2f} seconds")

# Run the script
if __name__ == "__main__":
    video_file_path = '/home/ec2-user/tiktok-extractor-v2/files/video/7185551271389072682.mp4'  # Replace with your video file path
    lang = 'eng'
    main(video_file_path, lang=lang)