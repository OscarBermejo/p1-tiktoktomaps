import boto3
import subprocess
import time
import botocore.exceptions
import os
import uuid
from moviepy.editor import VideoFileClip
import asyncio
import aioboto3

# AWS configuration
aws_region = 'eu-central-1'  # Change to your AWS region
kinesis_stream_name = 'p1-tiktoktomaps'  # Define a single Kinesis Video Stream name

# Initialize AWS clients
kinesis_client = boto3.client('kinesisvideo', region_name=aws_region)
rekognition_client = boto3.client('rekognition', region_name=aws_region)

# Initialize S3 client
s3_client = boto3.client('s3', region_name=aws_region)

def create_kinesis_stream_if_not_exists(stream_name):
    try:
        # Check if the stream already exists
        kinesis_client.describe_stream(StreamName=stream_name)
        print(f"Kinesis Video Stream '{stream_name}' already exists.")
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            # Create the stream if it does not exist
            kinesis_client.create_stream(
                StreamName=stream_name,
                DataRetentionInHours=24  # Adjust retention as needed
            )
            print(f"Created Kinesis Video Stream: {stream_name}")
            # Wait for the stream to become active
            while True:
                response = kinesis_client.describe_stream(StreamName=stream_name)
                if response['StreamInfo']['Status'] == 'ACTIVE':
                    print(f"Stream '{stream_name}' is now active.")
                    break
                time.sleep(1)
        else:
            raise

def get_data_endpoint(stream_name, api_name):
    response = kinesis_client.get_data_endpoint(
        StreamName=stream_name,
        APIName=api_name
    )
    return response['DataEndpoint']

def upload_video_to_kinesis(video_file_path, stream_name):
    # Convert video to a supported format using ffmpeg
    #converted_video_path = 'converted_video.mp4'
    converted_video_path = f"/home/ec2-user/tiktok-extractor-v2/files/video/converted_7185551271389072682.mp4"
    command = [
        'ffmpeg',
        '-i', video_file_path,
        '-c:v', 'libx264',  # Convert to H.264 codec
        '-preset', 'fast',  # Use a fast preset for encoding
        '-crf', '23',  # Constant Rate Factor for quality
        '-c:a', 'aac',  # Use AAC for audio codec
        '-b:a', '128k',  # Set audio bitrate
        '-f', 'mp4',  # Convert to MP4 format
        converted_video_path
    ]
    subprocess.run(command, check=True)

    # Upload the converted video to Kinesis Video Streams
    data_endpoint = get_data_endpoint(stream_name, 'PUT_MEDIA')
    print(f"Data Endpoint for Upload: {data_endpoint}")

    # Use ffmpeg to send video to Kinesis Video Streams
    command = [
        'ffmpeg',
        '-v', 'debug',  # Increase verbosity for debugging
        '-re',  # Read input at native frame rate
        '-i', converted_video_path,
        '-f', 'matroska',  # Use Matroska format for testing
        'output_test.mkv'  # Output to a local file for testing
    ]
    subprocess.run(command, check=True)

def upload_video_to_s3(video_file_path, bucket_name, object_name):
    try:
        s3_client.upload_file(video_file_path, bucket_name, object_name)
        print(f"Uploaded video to S3: s3://{bucket_name}/{object_name}")
    except Exception as e:
        print(f"Failed to upload video to S3: {e}")
        raise

def start_text_detection_s3(bucket_name, object_name, video_id):
    unique_token = f"{video_id}-{uuid.uuid4()}"
    response = rekognition_client.start_text_detection(
        Video={
            'S3Object': {
                'Bucket': bucket_name,
                'Name': object_name
            }
        },
        ClientRequestToken=unique_token
    )
    return response['JobId']

def get_text_detection_results(job_id):
    while True:
        response = rekognition_client.get_text_detection(JobId=job_id)
        status = response['JobStatus']
        print(f"Job Status: {status}")
        if status == 'SUCCEEDED':
            return response['TextDetections']
        elif status == 'FAILED':
            raise Exception(f"Text detection job failed: {response.get('StatusMessage', 'No status message')}")
        else:
            time.sleep(5)  # Wait before checking again

def convert_video_for_rekognition(input_path):
    output_path = input_path.rsplit('.', 1)[0] + '_converted.mp4'
    command = [
        'ffmpeg',
        '-i', input_path,
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-movflags', '+faststart',
        '-vf', 'scale=\'min(1920,iw)\':\'min(1080,ih)\':force_original_aspect_ratio=decrease',
        output_path
    ]
    subprocess.run(command, check=True)
    return output_path

def get_video_length(video_path):
    clip = VideoFileClip(video_path)
    duration = clip.duration
    clip.close()
    return duration

def main():
    start_time = time.time()
    
    video_file_path = '/home/ec2-user/tiktok-extractor-v2/files/video/7185551271389072682.mp4'
    video_id = '7185551271389072682'
    bucket_name = 'p1-tiktoktomaps'
    
    # Get video length
    video_length = get_video_length(video_file_path)
    
    # Convert video to a format supported by Rekognition
    converted_video_path = convert_video_for_rekognition(video_file_path)
    object_name = f'videos/{os.path.basename(converted_video_path)}'

    # Ensure the Kinesis Video Stream exists
    create_kinesis_stream_if_not_exists(kinesis_stream_name)

    # Upload converted video to S3
    upload_video_to_s3(converted_video_path, bucket_name, object_name)

    # Start text detection using S3
    job_id = start_text_detection_s3(bucket_name, object_name, video_id)
    print(f"Text detection job started with ID: {job_id}")

    # Get text detection results
    text_detections = get_text_detection_results(job_id)
    print("Extracted Texts:")
    for detection in text_detections:
        print(detection['TextDetection']['DetectedText'])

    # Clean up the converted video file
    os.remove(converted_video_path)

    end_time = time.time()
    total_processing_time = end_time - start_time

    print(f"\nTotal processing time: {total_processing_time:.2f} seconds")
    print(f"Video length: {video_length:.2f} seconds")

if __name__ == "__main__":
    main()
