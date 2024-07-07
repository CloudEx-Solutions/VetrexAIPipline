import os
import cv2
from google.cloud import storage
import google.generativeai as genai
from google.api_core import retry, exceptions
import json
import logging
from Tags_config import OBJECTS_OF_INTEREST
from Scope import SOURCE_BUCKET, DESTINATION_BUCKET, PROJECT_ID, LOCATION
import time
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

storage_client = storage.Client()

# Initialize Gemini API
genai.configure(api_key='YOUR_GEMINI_API_KEY')  # Replace with your actual API key
model = genai.GenerativeModel('gemini-pro-vision')

def encode_image(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

@retry.Retry(predicate=retry.if_exception_type(exceptions.PermissionDenied))
def analyze_frame(frame, frame_number):
    logger.info(f"Analyzing frame {frame_number}")
    
    encoded_image = encode_image(frame)
    
    prompt = f"""
    Analyze this image and identify objects, vehicles, buildings, and people.
    Focus on detecting: {', '.join(OBJECTS_OF_INTEREST)}.
    Provide a list of detected items with their confidence levels (high, medium, low).
    Format the output as a JSON string with 'object' and 'confidence' keys.
    """
    
    try:
        response = model.generate_content([prompt, encoded_image])
        logger.info(f"Raw Gemini response for frame {frame_number}: {response.text}")
        
        # Parse the JSON string from the response
        detections = json.loads(response.text)
        logger.info(f"Detections in frame {frame_number}: {detections}")
        return detections
    except Exception as e:
        logger.error(f"Error analyzing frame {frame_number}: {str(e)}", exc_info=True)
        return []

@retry.Retry(predicate=retry.if_exception_type(exceptions.PermissionDenied))
def process_video(video_name):
    logger.info(f"Processing video: {video_name}")
    try:
        source_bucket = storage_client.bucket(SOURCE_BUCKET)
        blob = source_bucket.blob(video_name)
        local_video_path = os.path.join(os.getcwd(), video_name)
        blob.download_to_filename(local_video_path)
        logger.info(f"Video downloaded to: {local_video_path}")

        cap = cv2.VideoCapture(local_video_path)
        if not cap.isOpened():
            raise Exception(f"Error opening video file: {local_video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        logger.info(f"Video info: {frame_count} frames, {fps} fps")
        
        all_detections = []
        frames_processed = 0
        
        for i in range(0, frame_count, 60):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame_detections = analyze_frame(frame, i)
                all_detections.extend(frame_detections)
                frames_processed += 1
            else:
                logger.warning(f"Could not read frame at position {i}")
        
        cap.release()

        logger.info(f"Processed {frames_processed} frames")

        # Clean up the local file
        os.remove(local_video_path)

        # Remove duplicates and sort by confidence
        unique_detections = {}
        for detection in all_detections:
            obj = detection['object']
            conf = detection['confidence']
            if obj not in unique_detections or conf > unique_detections[obj]['confidence']:
                unique_detections[obj] = detection

        final_detections = list(unique_detections.values())
        final_detections.sort(key=lambda x: ['low', 'medium', 'high'].index(x['confidence']), reverse=True)
        
        logger.info(f"Final detections: {final_detections}")
        
        output = {
            'video_name': video_name,
            'detections': final_detections
        }
        
        dest_bucket = storage_client.bucket(DESTINATION_BUCKET)
        json_blob = dest_bucket.blob(f'{video_name}.json')
        json_blob.upload_from_string(json.dumps(output, indent=2), content_type='application/json')
        logger.info(f"Results uploaded to: gs://{DESTINATION_BUCKET}/{video_name}.json")
    except Exception as e:
        logger.error(f"Error in process_video: {str(e)}", exc_info=True)
        raise

def main():
    logger.info("Starting main function")
    try:
        video_name = input("Enter the name of the video file to process (including extension): ")
        process_video(video_name)
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}", exc_info=True)

if __name__ == '__main__':
    main()