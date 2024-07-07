# video_analysis.py

import os
from google.cloud import storage
from google.cloud import videointelligence
import json
import logging
from Tags_config import OBJECTS_OF_INTEREST
from Scope import SOURCE_BUCKET, DESTINATION_BUCKET, PROJECT_ID, LOCATION

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

storage_client = storage.Client()
video_client = videointelligence.VideoIntelligenceServiceClient()

def matches_interest(detected_object, interests):
    detected_object = detected_object.upper()
    return any(interest in detected_object or detected_object in interest for interest in interests)

def analyze_video(video_uri):
    logger.info(f"Analyzing video: {video_uri}")
    try:
        features = [videointelligence.Feature.OBJECT_TRACKING, videointelligence.Feature.LABEL_DETECTION]
        request = videointelligence.AnnotateVideoRequest(
            input_uri=video_uri,
            features=features,
            location_id=LOCATION
        )

        operation = video_client.annotate_video(request=request)
        logger.info("Waiting for operation to complete...")
        result = operation.result(timeout=300)

        detected_objects = {}
        
        # Process object tracking results
        for annotation in result.annotation_results[0].object_annotations:
            object_name = annotation.entity.description.upper()
            confidence = annotation.confidence
            if matches_interest(object_name, OBJECTS_OF_INTEREST):
                if object_name not in detected_objects or confidence > detected_objects[object_name]:
                    detected_objects[object_name] = confidence
        
        # Process label detection results
        for label in result.annotation_results[0].segment_label_annotations:
            label_name = label.entity.description.upper()
            confidence = label.segments[0].confidence
            if matches_interest(label_name, OBJECTS_OF_INTEREST):
                if label_name not in detected_objects or confidence > detected_objects[label_name]:
                    detected_objects[label_name] = confidence

        detected_list = [{"object": obj, "confidence": conf} for obj, conf in detected_objects.items()]
        detected_list.sort(key=lambda x: x['confidence'], reverse=True)
        
        logger.info(f"Detected objects of interest: {detected_list}")
        return detected_list
    except Exception as e:
        logger.error(f"Error in analyze_video: {str(e)}")
        raise

def process_video(video_name):
    logger.info(f"Processing video: {video_name}")
    try:
        video_uri = f"gs://{SOURCE_BUCKET}/{video_name}"
        detected_objects = analyze_video(video_uri)
        
        output = {
            'video_name': video_name,
            'detected_objects': detected_objects
        }
        
        dest_bucket = storage_client.bucket(DESTINATION_BUCKET)
        json_blob = dest_bucket.blob(f'{video_name}.json')
        json_blob.upload_from_string(json.dumps(output, indent=2), content_type='application/json')
        logger.info(f"Results uploaded to: gs://{DESTINATION_BUCKET}/{video_name}.json")
    except Exception as e:
        logger.error(f"Error in process_video: {str(e)}")
        raise

def main():
    logger.info("Starting main function")
    try:
        video_name = input("Enter the name of the video file to process (including extension): ")
        process_video(video_name)
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")

if __name__ == '__main__':
    main()