import os
import cv2
from google.cloud import storage, vision
from google.api_core import retry, exceptions
import json
import logging
from Tags_config import OBJECTS_OF_INTEREST
from Scope import SOURCE_BUCKET, DESTINATION_BUCKET, PROJECT_ID, LOCATION
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

storage_client = storage.Client()
vision_client = vision.ImageAnnotatorClient()

def matches_interest(detected_object, interests):
    detected_object = detected_object.upper()
    return any(interest in detected_object or detected_object in interest for interest in interests)

@retry.Retry(predicate=retry.if_exception_type(exceptions.PermissionDenied))
def analyze_frame(frame, frame_number):
    logger.info(f"Analyzing frame {frame_number}")
    success, buffer = cv2.imencode('.jpg', frame)
    content = buffer.tobytes()
    image = vision.Image(content=content)
    
    try:
        object_response = vision_client.object_localization(image=image)
        label_response = vision_client.label_detection(image=image)
        
        logger.info(f"Raw object localization response for frame {frame_number}: {object_response}")
        logger.info(f"Raw label detection response for frame {frame_number}: {label_response}")
        
        objects = object_response.localized_object_annotations
        labels = label_response.label_annotations
        
        logger.info(f"Detected {len(objects)} objects and {len(labels)} labels in frame {frame_number}")

        all_detections = {}
        for obj in objects:
            all_detections[obj.name.upper()] = obj.score
        
        for label in labels:
            all_detections[label.description.upper()] = label.score

        logger.info(f"All detected objects/labels in frame {frame_number}: {all_detections}")
        return all_detections
    except Exception as e:
        logger.error(f"Error analyzing frame {frame_number}: {str(e)}", exc_info=True)
        return {}

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
        
        all_detections = {}
        frames_processed = 0
        
        for i in range(0, frame_count, 60):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame_detections = analyze_frame(frame, i)
                for obj, conf in frame_detections.items():
                    if obj not in all_detections or conf > all_detections[obj]:
                        all_detections[obj] = conf
                frames_processed += 1
            else:
                logger.warning(f"Could not read frame at position {i}")
        
        cap.release()

        logger.info(f"Processed {frames_processed} frames")

        # Clean up the local file
        os.remove(local_video_path)

        detected_list = [{"object": obj, "confidence": conf} for obj, conf in all_detections.items()]
        detected_list.sort(key=lambda x: x['confidence'], reverse=True)
        
        logger.info(f"All detections: {detected_list}")
        
        output = {
            'video_name': video_name,
            'detections': detected_list
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