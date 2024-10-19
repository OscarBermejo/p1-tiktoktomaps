import cv2
import pytesseract
import time

def extract_text_from_video(video_path, sample_interval=0.5):
    start_time = time.time()
    
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    frame_interval = int(fps * sample_interval)
    
    extracted_text = set()
    
    for frame_id in range(0, total_frames, frame_interval):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = video.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        extracted_text.update(text.split())
    
    video.release()
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"Video duration: {duration:.2f} seconds")
    print(f"Processing time: {processing_time:.2f} seconds")
    print("Extracted unique words:")
    print(extracted_text)

if __name__ == "__main__":
    video_path = '/home/ec2-user/tiktok-extractor-v2/files/video/7185551271389072682.mp4'
    extract_text_from_video(video_path)