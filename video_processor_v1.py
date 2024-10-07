import cv2
import numpy as np
import pytesseract
from PIL import Image
from functools import lru_cache
import asyncio
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import logger_config 
import os
import boto3
import time
import urllib.request
from botocore.exceptions import ClientError
import json
import psutil
from concurrent.futures import ThreadPoolExecutor
import re
import subprocess

logger = logger_config.get_logger(__name__)


def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(gray, 5, 75, 75)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Apply morphological operations to remove small noise
    kernel = np.ones((2, 2), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    return morph

def download_east_model():
    model_url = "https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb"
    model_path = "frozen_east_text_detection.pb"
    
    if not os.path.exists(model_path):
        print("Downloading EAST text detection model...")
        urllib.request.urlretrieve(model_url, model_path)
        print("Download completed.")

def extract_text_from_image(image, frame_number):
    
    # Ensure the EAST model is available
    download_east_model()
    
    # Parameters for EAST text detector
    east_model = 'frozen_east_text_detection.pb'
    conf_threshold = 0.5
    nms_threshold = 0.4

    # Load the pre-trained EAST model
    net = cv2.dnn.readNet(east_model)
    
    # Prepare the image
    orig = image.copy()
    (H, W) = image.shape[:2]
    newW, newH = (320, 320)
    rW = W / float(newW)
    rH = H / float(newH)
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]
    
    # Define the output layers for the EAST detector
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]
    
    # Construct a blob from the image and perform a forward pass
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                    (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    
    # Decode the predictions, then apply non-maxima suppression
    rectangles, confidences = decode_predictions(scores, geometry, conf_threshold)
    indices = cv2.dnn.NMSBoxesRotated(rectangles, confidences, conf_threshold, nms_threshold)
    
    result_text = ""

    try:
        # Check if any text regions are detected
        if len(indices) > 0:
            # Loop over the indices to extract text regions
            for i in indices:
                # Check if i is an array of indices
                if isinstance(i, (list, tuple)):
                    i = i[0]
                
                # Get the rotated rectangle parameters
                vertices = cv2.boxPoints(rectangles[i])
                vertices = np.int0(vertices * [rW, rH])
                
                # Extract the region of interest (ROI)
                roi = orig[vertices[1][1]:vertices[3][1], vertices[1][0]:vertices[3][0]]
                
                # Preprocess the ROI
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                
                # Apply OCR on the ROI
                text = pytesseract.image_to_string(roi, config='--psm 7')
                text = re.sub(r'[^a-zA-Z0-9\s.,\'-]', '', text).strip()
                
                if text:
                    result_text += ' ' + text
        
        # If no text regions are detected, fall back to advanced OCR
        if not result_text:
            logger.warning("EAST detection failed. Falling back to advanced OCR.")
            result_text = advanced_ocr(orig)
        
        return result_text

    except Exception as e:
        logger.error(f"Error in extract_text_from_image: {str(e)}")
        logger.warning(f"EAST detection failed. Falling back to advanced OCR. Excpetion {e}")
        return advanced_ocr(image)


def decode_predictions(scores, geometry, conf_threshold):
    # Retrieve the number of rows and columns from the scores volume
    (numRows, numCols) = scores.shape[2:4]
    rectangles = []
    confidences = []

    for y in range(numRows):
        # Extract the scores and geometrical data
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        
        for x in range(numCols):
            score = scoresData[x]
            if score < conf_threshold:
                continue

            # Compute the offset factor
            offsetX = x * 4.0
            offsetY = y * 4.0

            # Calculate the rotation angle
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # Calculate the width and height
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # Calculate the coordinates
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # Append the rectangle and confidence score
            rect = ((startX + w / 2, startY + h / 2), (w, h), -1 * np.degrees(angle))
            rectangles.append(rect)
            confidences.append(float(score))

    return rectangles, confidences

def is_text_in_frame(frame):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        text = pytesseract.image_to_string(thresh)
        return bool(text.strip())
    except Exception as e:
        logger.error(f"Error in is_text_in_frame: {str(e)}")
        return False

def process_frame_batch(video_file, start, end, frame_skip):
    try:
        cap = cv2.VideoCapture(video_file)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        texts = []
        
        logger.info(f"Starting to process batch from frame {start} to {end} with frame_skip {frame_skip}")
        
        # Create a directory to store processed frames
        os.makedirs('processed_frames', exist_ok=True)
        
        frames_processed = []
        current_frame = start
        while current_frame < end:
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Failed to read frame {current_frame}")
                break
            
            if (current_frame - start) % frame_skip == 0:
                frames_processed.append(current_frame)
                
                # Extract text from the frame
                text = extract_text_from_image(frame, current_frame)
                
                if text:
                    texts.append((current_frame, text))
                    logger.debug(f"Text found in frame {current_frame}: {text}")
                else:
                    logger.debug(f"No text found in frame {current_frame}")
                
                # Save the processed frame
                frame_filename = f'processed_frames/frame_{current_frame:06d}.jpg'
                cv2.imwrite(frame_filename, frame)
                logger.debug(f"Saved processed frame: {frame_filename}")
            
            current_frame += 1
        
        cap.release()
        
        logger.info(f"Batch processing completed. Processed frames: {frames_processed}")
        logger.info(f"Found text in {len(texts)} frames out of {len(frames_processed)} processed")
        
        return texts
    except Exception as e:
        logger.error(f"Error in process_frame_batch: {str(e)}", exc_info=True)
        return []

async def extract_text_from_video(video_file, num_workers=3):
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = max(1, int(fps / 2))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    batch_size = total_frames // num_workers
    batches = [(i * batch_size, min((i + 1) * batch_size, total_frames)) for i in range(num_workers)]

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [loop.run_in_executor(executor, process_frame_batch, video_file, start, end, frame_skip) 
                   for start, end in batches]
        results = await asyncio.gather(*futures)

    all_results = [item for sublist in results for item in sublist]
    all_results.sort(key=lambda x: x[0])
    return all_results

def amazon_audio_transcribe(audio_file_path, job_name, language_code='en-US'):
    """
    Transcribe the given audio file using Amazon Transcribe.
    
    :param audio_file_path: Path to the local audio file
    :param job_name: Unique name for the transcription job
    :param language_code: Language code for the audio (default is 'en-US')
    :return: Transcription text or error message
    """
    # Initialize the Transcribe client
    transcribe = boto3.client('transcribe', region_name = 'eu-central-1')

    # Upload the audio file to S3
    s3 = boto3.client('s3')
    bucket_name = 'p1-tiktoktomaps'  # Replace with your S3 bucket name
    s3_audio_path = f'transcribe_audio/{os.path.basename(audio_file_path)}'
    
    try:
        s3.upload_file(audio_file_path, bucket_name, s3_audio_path)
    except ClientError as e:
        logger.error(f"Error uploading file to S3: {str(e)}")
        return f"Error uploading file to S3: {str(e)}"

    # Start the transcription job
    job_uri = f's3://{bucket_name}/{s3_audio_path}'
    try:
        transcribe.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': job_uri},
            MediaFormat='wav',  # Adjust this based on your audio format
            LanguageCode=language_code
        )

        # Wait for the job to complete
        while True:
            status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
            if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
                break
            logger.info("Transcription job still in progress...")
            time.sleep(5)

        if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
            response = urllib.request.urlopen(status['TranscriptionJob']['Transcript']['TranscriptFileUri'])
            data = response.read()
            text = data.decode('utf-8')
            text_data = json.loads(data)

            # Extract just the transcript text
            transcript = text_data['results']['transcripts'][0]['transcript']
            
            # Clean up: delete the transcription job and S3 file
            transcribe.delete_transcription_job(TranscriptionJobName=job_name)
            s3.delete_object(Bucket=bucket_name, Key=s3_audio_path)
            
            return transcript
        else:
            error_reason = status['TranscriptionJob'].get('FailureReason', 'Unknown error')
            logger.error(f"Transcription job failed: {error_reason}")
            return f"Transcription failed: {error_reason}"

    except Exception as e:
        logger.error(f"Error in transcription process: {str(e)}")
        return f"Error in transcription process: {str(e)}"

async def process_audio(video_file):
    logger.info(f'Extracting and processing audio from mp4 file')
    video = VideoFileClip(video_file)
    audio = video.audio
    audio_file = video_file.replace('video', 'audio').replace('.mp4', '.wav')
    await asyncio.get_event_loop().run_in_executor(
        None, 
        lambda: audio.write_audiofile(audio_file, fps=44100, codec='pcm_s16le')
    )
    return audio_file

async def process_audio_v2(video_file):
    try:
        logger.info(f'Extracting audio from video file: {video_file}')
        audio_file = video_file.replace('video', 'audio').replace('.mp4', '.wav')

        # Build the FFmpeg command
        # We use '-vn' to skip the video stream, '-acodec pcm_s16le' for WAV format
        command = [
            'ffmpeg',
            '-y',  # Overwrite output files without asking
            '-i', video_file,
            '-vn',  # Disable video recording
            '-acodec', 'pcm_s16le',  # Audio codec
            '-ar', '44100',  # Set audio sampling rate
            '-ac', '2',  # Set number of audio channels
            audio_file
        ]

        # Run the command asynchronously
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )

        await process.communicate()

        if process.returncode != 0:
            logger.error(f'FFmpeg exited with return code {process.returncode}')
            return None

        logger.info(f'Audio extracted to: {audio_file}')
        return audio_file

    except Exception as e:
        logger.error(f"Error in process_audio: {str(e)}", exc_info=True)
        return None


async def fallback_google_transcribe(audio_file):
    logger.info(f'Transcribing audio file: {audio_file}')
    job_name = f"transcribe_job_{int(time.time())}"  # Create a unique job name
    transcription = amazon_audio_transcribe(audio_file, job_name)
    if transcription.startswith("Error") or transcription.startswith("Transcription failed"):
        logger.warning(f"Amazon Transcribe failed: {transcription}")
        logger.info("Falling back to Google Speech Recognition")
        return await fallback_google_transcribe(audio_file)
    return transcription

async def transcribe_audio(audio_file):
    logger.info(f'Transcribing audio file: {audio_file}')
    job_name = f"transcribe_job_{int(time.time())}"  # Create a unique job name
    transcription = amazon_audio_transcribe(audio_file, job_name)
    if transcription.startswith("Error") or transcription.startswith("Transcription failed"):
        logger.warning(f"Amazon Transcribe failed: {transcription}")
        logger.info("Falling back to Google Speech Recognition")
        return await fallback_google_transcribe(audio_file)
    return transcription


async def analyze_video_content(video_file):
    try:
        # Get video properties
        cap = cv2.VideoCapture(video_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        cap.release()

        logger.info(f"Video properties: {total_frames} frames, {fps} fps, {duration:.2f} seconds")

        # Calculate frame_skip to process 2 frames per second
        frame_skip = max(1, int(fps / 2))
        
        # Define batch size (e.g., 5 seconds worth of frames)
        batch_size = int(5 * fps)

        # Calculate batch ranges
        batch_ranges = [(i, min(i + batch_size, total_frames)) 
                        for i in range(0, total_frames, batch_size)]

        logger.info(f"Processing {len(batch_ranges)} batches, each covering {batch_size/fps:.2f} seconds, frame_skip: {frame_skip}")

        # Process video frames in batches
        futures = [asyncio.to_thread(process_frame_batch, video_file, start, end, frame_skip)
                   for start, end in batch_ranges]
        
        results = []
        for i, future in enumerate(asyncio.as_completed(futures)):
            try:
                result = await future
                results.extend(result)
                start, end = batch_ranges[i]  # Get the start and end for this batch
                logger.info(f"Completed batch {i+1}/{len(batch_ranges)}, frames {start} to {end}")
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}", exc_info=True)
        
        logger.info(f"All batches completed. Processed a total of {len(results)} frames with text")

        # Process audio
        audio_file = await process_audio_v2(video_file)
        audio_transcription = await transcribe_audio(audio_file)

        logger.info("Video and audio processing completed")

        return results, audio_transcription
    
    except Exception as e:
        logger.error(f"Error in analyze_video_content: {str(e)}", exc_info=True)
        raise
    
    finally:
        # Clean up
        if os.path.exists(video_file):
            try:
                os.remove(video_file)
                os.remove(video_file.replace('video', 'audio').replace('.mp4', '.wav'))
                logger.info(f"Deleted video and audio files: {video_file}")
            except Exception as e:
                logger.error(f"Error deleting video file {video_file}: {str(e)}")
    

# Usage
async def process_video(video_file):
    try:
        logger.info(f"Starting video processing for file: {video_file}")
        texts, audio_transcription = await analyze_video_content(video_file)
        logger.info("Video processing completed successfully")
        
        return {
            'texts': texts if texts else [],  # Ensure this is always a list
            'audio_transcription': audio_transcription if audio_transcription else "No transcription"
        }
    except Exception as e:
        logger.error(f"Error in video processing for file {video_file}: {str(e)}", exc_info=True)
        return None

def get_executor():
    return ThreadPoolExecutor(max_workers=4)  # Adjust the number of workers as needed

def log_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

def advanced_ocr(image, min_height=30):
    try:
        # Preprocess the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on height
        large_contours = [cnt for cnt in contours if cv2.boundingRect(cnt)[3] > min_height]
        
        # Create a mask for large contours
        mask = np.zeros_like(thresh)
        cv2.drawContours(mask, large_contours, -1, 255, -1)
        
        # Apply OCR with custom configuration
        config = r'--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ '
        text = pytesseract.image_to_string(mask, config=config)
        
        # Postprocess the extracted text
        text = re.sub(r'[^A-Z\s]', '', text).strip()
        text = ' '.join(text.split())
        
        return text if len(text) > 0 else ""
    except Exception as e:
        logger.error(f"Error in advanced_ocr: {str(e)}")
        return ""