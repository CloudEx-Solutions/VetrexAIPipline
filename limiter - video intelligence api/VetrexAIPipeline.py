import os
import cv2
from google.cloud import storage, vision
import json
import logging
from Tags_config import OBJECTS_OF_INTEREST
from Scope import SOURCE_BUCKET, DESTINATION_BUCKET, PROJECT_ID, LOCATION
from google.api_core import retry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

storage_client = storage.Client()
vision_client = vision.ImageAnnotatorClient()

def matches_interest(detected_object, interests):
    detected_object = detected_object.upper()
    return any(interest in detected_object or detected_object in interest for interest in interests)



@retry.Retry(predicate=retry.if_exception_type(google.api_core.exceptions.PermissionDenied))
def process_video(video_name):
    logger.info(f"Processing video: {video_name}")
    try:
        # Download video from source bucket
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
        
        interval = max(1, fps * 5)  # Analyze a frame every 5 seconds
        
        all_detections = {}
        
        for i in range(0, frame_count, interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame_detections = analyze_frame(frame)
                for obj, conf in frame_detections.items():
                    if obj not in all_detections or conf > all_detections[obj]:
                        all_detections[obj] = conf
            else:
                logger.warning(f"Could not read frame at position {i}")
        
        cap.release()

        # Clean up the local file
        os.remove(local_video_path)

        detected_list = [{"object": obj, "confidence": conf} for obj, conf in all_detections.items()]
        detected_list.sort(key=lambda x: x['confidence'], reverse=True)
        
        output = {
            'video_name': video_name,
            'detected_objects': detected_list
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