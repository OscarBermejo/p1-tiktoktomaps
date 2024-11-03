import cv2
import boto3
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from botocore.exceptions import NoCredentialsError
import pytesseract

# AWS configuration
aws_region = 'eu-central-1'  # Change to your AWS region
s3_bucket_name = 'p1-tiktoktomaps'  # Replace with your S3 bucket name

# Initialize AWS clients
s3_client = boto3.client('s3', region_name=aws_region)
textract_client = boto3.client('textract', region_name=aws_region)

def upload_to_s3(file_path, bucket_name, object_name):
    try:
        print(f"Uploading {file_path} to S3 bucket {bucket_name} as {object_name}...")
        s3_client.upload_file(file_path, bucket_name, object_name)
        print(f"Uploaded {object_name} to S3.")
    except NoCredentialsError:
        print("AWS credentials not available.")

def extract_text_from_image(s3_bucket, s3_object):
    try:
        print(f"Extracting text from {s3_object} using AWS Textract...")
        response = textract_client.detect_document_text(
            Document={'S3Object': {'Bucket': s3_bucket, 'Name': s3_object}}
        )
        text = ''
        for item in response['Blocks']:
            if item['BlockType'] == 'LINE':
                text += item['Text'] + '\n'
        print(f"Extracted text from {s3_object}.")
        return text
    except Exception as e:
        print(f"Error in Textract for {s3_object}: {e}")
        return ''

def contains_text(image, min_conf=60):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Use pytesseract to detect text
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    for conf in data['conf']:
        if int(conf) > min_conf:
            return True
    return False

def extract_frames(video_path, output_folder, frame_interval):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    cap = cv2.VideoCapture(video_path)
    current_frame = 0
    frame_paths = []

    print(f"Extracting frames from {video_path} every {frame_interval} frames...")
    while True:
        ret = cap.grab()
        if not ret:
            break

        if current_frame % frame_interval == 0:
            ret, frame = cap.retrieve()
            if not ret:
                break

            # Check if the frame contains text
            if contains_text(frame):
                frame_filename = os.path.join(output_folder, f"frame{current_frame}.jpg")
                cv2.imwrite(frame_filename, frame)
                frame_paths.append(frame_filename)
                print(f"Extracted frame {current_frame} to {frame_filename}")

        current_frame += 1

    cap.release()
    print("Finished extracting frames.")
    return frame_paths

def process_video(video_path, video_id, frame_interval=30):
    print(f"Processing video: {video_path}")
    output_folder = f'frames/{video_id}/'
    frame_paths = extract_frames(video_path, output_folder, frame_interval)

    s3_objects = []
    for frame_path in frame_paths:
        # Include video_id in the S3 object name
        s3_object_name = f"frames/{video_id}/{os.path.basename(frame_path)}"
        upload_to_s3(frame_path, s3_bucket_name, s3_object_name)
        s3_objects.append(s3_object_name)

    print("Starting text extraction from images...")
    extracted_texts = []

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_s3_object = {executor.submit(extract_text_from_image, s3_bucket_name, s3_object): s3_object for s3_object in s3_objects}
        for future in as_completed(future_to_s3_object):
            s3_object = future_to_s3_object[future]
            try:
                text = future.result()
                extracted_texts.append(text)
            except Exception as e:
                print(f"Error processing {s3_object}: {e}")

    # Clean up local frames
    print("Cleaning up local frame files...")
    for frame_path in frame_paths:
        os.remove(frame_path)
        print(f"Deleted {frame_path}")

    print("Finished processing video.")
    return extracted_texts

if __name__ == "__main__":
    start_time = time.time()  # Start timing

    video_file_path = '/home/ec2-user/tiktok-extractor-v2/files/video/7185551271389072682.mp4' 
    video_id = '7185551271389072682'
    extracted_texts = process_video(video_file_path, video_id)
    
    print("Extracted Texts:")
    for text in extracted_texts:
        print(text)

    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
