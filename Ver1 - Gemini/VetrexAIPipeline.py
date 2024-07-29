import os
import cv2
from google.cloud import storage
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models
from google.api_core import retry, exceptions
import json
import logging
import base64
from Tags_config import OBJECTS_OF_INTEREST
from Scope import SOURCE_BUCKET, DESTINATION_BUCKET, PROJECT_ID, LOCATION
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

storage_client = storage.Client()

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel("gemini-1.5-flash-001")

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

def encode_image(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def clean_response_text(response_text):
    """Remove Markdown formatting and any unexpected characters from the response text."""
    if response_text.startswith("```json") and response_text.endswith("```"):
        response_text = response_text[7:-3].strip()
    response_text = response_text.replace('\n', ' ').strip()
    return response_text

@retry.Retry(predicate=retry.if_exception_type(exceptions.PermissionDenied))
def analyze_frame(frame, frame_number):
    logger.info(f"Analyzing frame {frame_number}")
    
    encoded_image = encode_image(frame)
    
    prompt = f"""
    Analyze this image and identify objects, vehicles, buildings, and people.
    Focus on detecting: {', '.join(OBJECTS_OF_INTEREST)}.
    Provide a list of detected items with their confidence levels (high, medium, low).
    Format the output as a JSON string with the following structure:
    {{
      "detected_objects": [
        {{"object": "<object_name>", "confidence": "<confidence_level>"}}
      ],
      "description": "<short_description_of_the_scene>"
    }}
    """
    
    try:
        responses = model.generate_content(
            [prompt, encoded_image],
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=True,
        )
        response_text = ''.join([response.text for response in responses])
        response_text = clean_response_text(response_text)
        
        logger.info(f"Raw Gemini response for frame {frame_number}: {response_text}")
        
        # Parse the JSON string from the response
        response_json = json.loads(response_text)
        detections = response_json.get("detected_objects", [])
        description = response_json.get('description', '')

        logger.info(f"Detections in frame {frame_number}: {detections}")
        logger.info(f"Description for frame {frame_number}: {description}")
        
        return detections, description
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON for frame {frame_number}: {response_text}", exc_info=True)
        return [], ""
    except Exception as e:
        logger.error(f"Error analyzing frame {frame_number}: {str(e)}", exc_info=True)
        return [], ""

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
        descriptions = []

        # Process frames concurrently
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(0, frame_count, 120):  # Increase the interval for fewer frames
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    futures.append(executor.submit(analyze_frame, frame, i))
                else:
                    logger.warning(f"Could not read frame at position {i}")

            for future in as_completed(futures):
                frame_detections, description = future.result()
                all_detections.extend(frame_detections)
                descriptions.append(description)
                frames_processed += 1
        
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

        # Combine descriptions into a single summary
        final_description = " ".join(set(descriptions)).strip()  # Use set to remove duplicate descriptions
        
        logger.info(f"Final detections: {final_detections}")
        logger.info(f"Final description: {final_description}")
        
        output = {
            'video_name': video_name,
            'detections': final_detections,
            'description': final_description
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
