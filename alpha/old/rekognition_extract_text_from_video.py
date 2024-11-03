import boto3
import time
import subprocess
import threading

# AWS configuration
aws_region = 'eu-central-1'  # Change to your AWS region
s3_bucket_name = 'p1-tiktoktomaps'  # Replace with your S3 bucket name

# Initialize AWS clients
s3_client = boto3.client('s3', region_name=aws_region)
rekognition_client = boto3.client('rekognition', region_name=aws_region)

def convert_video(input_path, output_path):
    command = [
        'ffmpeg',
        '-i', input_path,
        '-c:v', 'libx264',
        '-preset', 'fast',  # Use 'fast' preset for quicker conversion
        '-crf', '23',
        '-c:a', 'aac',
        '-b:a', '128k',
        output_path
    ]
    subprocess.run(command, check=True)

def upload_to_s3(file_path, bucket_name, object_name):
    try:
        print(f"Uploading {file_path} to S3 bucket {bucket_name} as {object_name}...")
        s3_client.upload_file(file_path, bucket_name, object_name)
        print(f"Uploaded {object_name} to S3.")
    except Exception as e:
        print(f"Error uploading {file_path} to S3: {e}")

def start_text_detection(bucket, video_name):
    response = rekognition_client.start_text_detection(
        Video={
            'S3Object': {
                'Bucket': bucket,
                'Name': video_name
            }
        }
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
            time.sleep(3)  # Reduced sleep time for quicker status checks

def process_video(video_path, video_id):
    # Convert video to supported format
    converted_video_path = f"/home/ec2-user/tiktok-extractor-v2/files/video/converted_{video_id}.mp4"
    convert_video(video_path, converted_video_path)
    
    print(f"Processing video: {converted_video_path}")
    s3_object_name = f"videos/{video_id}.mp4"

    # Start upload and detection in parallel
    upload_thread = threading.Thread(target=upload_to_s3, args=(converted_video_path, s3_bucket_name, s3_object_name))
    upload_thread.start()

    print("Starting text detection...")
    job_id = start_text_detection(s3_bucket_name, s3_object_name)
    print(f"Text detection job started with ID: {job_id}")

    # Wait for upload to complete
    upload_thread.join()

    print("Waiting for text detection results...")
    text_detections = get_text_detection_results(job_id)

    extracted_texts = [detection['TextDetection']['DetectedText'] for detection in text_detections]

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
