# gemini_integration.py
import os
import logging
import google.generativeai as genai
from PIL import Image
import time
import cv2
import re
from PIL import Image
import numpy as np
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_gemini():
    """
    Initializes the Gemini API with the API key from the environment variable.
    """
    try:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        logging.info("Gemini API initialized successfully.")
    except KeyError:
        logging.error("GEMINI_API_KEY environment variable not set.")
        raise

    try:
        model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")
        logging.info("Gemini model initialized successfully.")
        return model
    except Exception as e:
        logging.error(f"Failed to initialize Gemini model: {e}")
        raise


def process_with_video_model(model, video_path, user_query, collage_path):
    """
    Sends a query to the Gemini video model, requests JSON output, parses the JSON
    for timestamps, converts them to frames, and saves those frames to a collage.
    """
    logging.info("Processing with video model using Google Gemini API...")

    try:
        video_file = genai.upload_file(path=video_path)
    except Exception as e:
        logging.error(f"Failed to upload video file: {e}")
        return
    
    # Wait for the video processing to complete
    while video_file.state.name == "PROCESSING":
        time.sleep(10)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        logging.error("Video processing failed.")
        return
    # Generate content using the Gemini model
    prompt = f"""
    Find scenes in the video related to: "{user_query}".
    Return the relevant scenes in the following JSON format:
    {{
        "timestamps": [
            {{"start": "HH:MM:SS"}},
            ...
        ]
    }}
    Do not include and code blocks or markdown formatting.
    """

    try:
        response = model.generate_content([video_file, prompt], request_options={"timeout": 600})
        # Ensure response.text is parsed into a JSON object
        response_json = json.loads(response.text)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON response: {e}")
        return
    except Exception as e:
        logging.error(f"Failed to generate content: {e}")
        return
    except Exception as e:
        logging.error(f"Failed to generate content: {e}")
        return
    
    # Extract timestamps from the response
    timestamps = extract_timestamps(response_json)
    if not timestamps:
        logging.warning("No relevant scenes found.")
        return

    # Extract frames at the specified timestamps
    frames = extract_frames_at_timestamps(video_path, timestamps)
    if not frames:
        logging.warning("No frames extracted.")
        return

    # Create and save the collage
    create_collage_from_frames(frames, collage_path)
    logging.info(f"Collage saved as '{collage_path}'.")


def extract_timestamps(response_json):
    """
    Extracts timestamps from the JSON response.
    """
    timestamps = []
    for entry in response_json["timestamps"]:
        if "start" in entry:
            timestamps.append(entry["start"])
        else:
            logging.warning(f"Invalid timestamp entry: {entry}")
    return timestamps


def extract_frames_at_timestamps(video_path, timestamps):
    """
    Extracts frames from the video at the specified timestamps.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    for ts in timestamps:
        time_parts = list(map(int, ts.split(':')))
        if len(time_parts) == 2:  # Format is MM:SS
            m, s = time_parts
            h = 0
        elif len(time_parts) == 3:  # Format is HH:MM:SS
            h, m, s = time_parts
        else:
            logging.warning(f"Invalid timestamp format: {ts}")
            continue

        # Calculate the frame number from the timestamp
        total_seconds = h * 3600 + m * 60 + s
        frame_number = int(total_seconds * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            logging.warning(f"Failed to extract frame at {ts}.")

    cap.release()
    return frames


def create_collage_from_frames(frames, collage_path):
    """
    Creates a collage from the extracted frames and saves it to the specified path.
    """
    if not frames:
        logging.warning("No frames to create a collage.")
        return

    # Convert frames to PIL images
    images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]

    # Define collage parameters
    cols = min(5, len(images))
    rows = (len(images) + cols - 1) // cols
    thumb_width = 200
    thumb_height = int(thumb_width * images[0].height / images[0].width)

    collage_width = cols * thumb_width
    collage_height = rows * thumb_height

    collage_image = Image.new('RGB', (collage_width, collage_height), color=(255, 255, 255))

    for idx, img in enumerate(images):
        img = img.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        x = (idx % cols) * thumb_width
        y = (idx // cols) * thumb_height
        collage_image.paste(img, (x, y))

    collage_image.save(collage_path)
    collage_image.show()

