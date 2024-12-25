# video_search_engine.py
import os
import logging
from video_downloader import download_video
from image_model_processor import (
    detect_and_save_scene_frames,
    generate_captions,
    load_captions,
    search_with_autocomplete,
    create_collage
)
from video_model_processor import (
    initialize_gemini,
    process_with_video_model
)

# Initialize logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
SCENE_DIR = 'scene_frames'
CAPTIONS_FILE = 'scene_captions.json'
VIDEO_PATH = 'downloaded_video.mp4'
COLLAGE_PATH = 'collage.png'


def main():
    # download video if not already downloaded
    if not os.path.exists(VIDEO_PATH):
        video_file = download_video(VIDEO_PATH)
        if not video_file:
            logging.error("Failed to download video.")
            return
        
    # Ask user for processing mode
    mode = input("Choose mode:\n1. Search by scene captions\n2. Search using Gemini video model\nEnter 1 or 2: ").strip()   

    if mode == '1':
        # If captions file exists, skip download, creation of image, and captioning
        if not os.path.exists(CAPTIONS_FILE):
            # Ensure output directory for scene frames exists
            if not os.path.exists(SCENE_DIR):
                os.makedirs(SCENE_DIR)
                logging.info("Created output directory for scene frames.")
            
                # download video if not already downloaded
                if not os.path.exists(VIDEO_PATH):
                    video_file = download_video(VIDEO_PATH)
                    if not video_file:
                        logging.error("Failed to download video. Exiting.")
                        return
        
                # Detect scenes and save frames
                scene_list = detect_and_save_scene_frames(VIDEO_PATH, SCENE_DIR)
                if not scene_list:
                    logging.error("Failed to detect scenes. Exiting.")   
                    return
            
            # Initialize the Moondream2 model
            logging.info("Initializing Moondream2 model...")
            try:
                model = md.vl(model="./moondream-2b-int8.mf")
            except Exception as e:
                logging.error(f"Failed to initialize Moondream2 model: {e}. Exiting.")
                return
            logging.info("Moondream2 model initialized.")

            # Generate captions for the scenes
            scene_captions = generate_captions(SCENE_DIR, CAPTIONS_FILE, model)
        
        # Load scene captions from file
        scene_captions = load_captions(CAPTIONS_FILE)
        if not scene_captions:
            logging.error("Failed to load scene captions. Exiting.")
            return
        matching_scenes = search_with_autocomplete(scene_captions)
        if not matching_scenes:
            logging.warning("No matching scenes found. Exiting.")
            return
        create_collage(matching_scenes, SCENE_DIR, COLLAGE_PATH)
        if not os.path.exists('collage.png'):
            logging.warning("No collage created. Exiting.")
            return
    elif mode == '2':
        # Initialize the Gemini API
        try:
            model = initialize_gemini()
        except Exception as e:
            logging.error(f"Failed to initialize Gemini model: {e}. Exiting.")
            return
        
        # Prompt user for search query
        user_query = input("Using a video model. What would you like me to find in the video? ").strip()

        # Process video with Gemini model
        process_with_video_model(model, VIDEO_PATH, user_query, COLLAGE_PATH)

if __name__ == "__main__":
    main()
