import os
from google.cloud import storage
from google.cloud import videointelligence
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define bucket names and other constants
SOURCE_BUCKET = 'vertex_data_source'
DESTINATION_BUCKET = 'vertex_data_output'
PROJECT_ID = 'mod-gcp-white-soi-dev-1'
LOCATION = 'us-east1'

# Set up Google Cloud Storage client
storage_client = storage.Client()

# Set up Video Intelligence client
video_client = videointelligence.VideoIntelligenceServiceClient()

def analyze_video(video_uri):
    logger.info(f"Analyzing video: {video_uri}")
    try:
        # Configure the request
        features = [videointelligence.Feature.OBJECT_TRACKING]
        request = videointelligence.AnnotateVideoRequest(
            input_uri=video_uri,
            features=features,
            location_id=LOCATION
        )

        # Make the request
        operation = video_client.annotate_video(request=request)
        logger.info("Waiting for operation to complete...")
        result = operation.result(timeout=300)  # Adjust timeout as needed

        # Process results
        detected_objects = {}
        for annotation in result.annotation_results[0].object_annotations:
            object_name = annotation.entity.description.lower()
            confidence = annotation.confidence
            # Store the object with its highest confidence score
            if object_name not in detected_objects or confidence > detected_objects[object_name]:
                detected_objects[object_name] = confidence

        # Convert to list of dictionaries for easier JSON serialization
        object_list = [{"object": obj, "confidence": conf} for obj, conf in detected_objects.items()]
        
        # Sort by confidence score, highest first
        object_list.sort(key=lambda x: x['confidence'], reverse=True)
        
        logger.info(f"Detected objects: {object_list}")
        return object_list
    except Exception as e:
        logger.error(f"Error in analyze_video: {str(e)}")
        raise

def process_video(video_name):
    logger.info(f"Processing video: {video_name}")
    try:
        # Analyze video
        video_uri = f"gs://{SOURCE_BUCKET}/{video_name}"
        objects = analyze_video(video_uri)
        
        # Create output JSON
        output = {
            'video_name': video_name,
            'detected_objects': objects
        }
        
        # Upload JSON to destination bucket
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
        # List all videos in the source bucket
        source_bucket = storage_client.bucket(SOURCE_BUCKET)
        blobs = source_bucket.list_blobs()
        
        for blob in blobs:
            if blob.name.endswith('.mp4'):
                process_video(blob.name)
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")

if __name__ == '__main__':
    main()