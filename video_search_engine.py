# video_search_engine.py
import os
import logging
import yt_dlp
from image_model_processor import initialize_moondream_model, process_with_image_model
from video_model_processor import initialize_gemini, process_with_video_model
from colorama import Fore

# Initialize logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
# Paths
SCENE_DIR = 'scene_frames'
CAPTIONS_FILE = 'scene_captions.json'
VIDEO_PATH = 'downloaded_video.mp4'
COLLAGE_PATH = 'collage.png'


def main():
    # download video if not already downloaded
    if not os.path.exists(VIDEO_PATH):
        # Search for a "Super Mario movie trailer" on YouTube and download it.
        ydl_opts = {
            'format': 'bestvideo',
            'noplaylist': True,
            'outtmpl': VIDEO_PATH
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logging.info("Downloading video...")
                ydl.download(['ytsearch:super mario movie trailer'])
        except Exception as e:
            logging.error(Fore.RED + f"Failed to download video: {e}")
            return
        
    # Ask user for processing mode
    mode = input(Fore.BLUE + "\nChoose mode:\n1. Search by scene captions\n2. Search using Gemini video model\nEnter 1 or 2: \n").strip()   

    if mode == '1':
        try:
            model = initialize_moondream_model()
        except Exception as e:
            logging.error(Fore.RED + f"Failed to initialize Moondream2 model: {e}. Exiting.")
            return
        
        # Process with image model
        process_with_image_model(model, CAPTIONS_FILE, SCENE_DIR, COLLAGE_PATH, VIDEO_PATH)

    elif mode == '2':
        # Initialize the Gemini API
        try:
            model = initialize_gemini()
        except Exception as e:
            logging.error(Fore.RED + f"Failed to initialize Gemini model: {e}. Exiting.")
            return
        
        # Prompt user for search query
        user_query = input(Fore.GREEN + "\nUsing a video model. What would you like me to find in the video? \n").strip()

        # Process video with Gemini model
        process_with_video_model(model, VIDEO_PATH, user_query, COLLAGE_PATH)

if __name__ == "__main__":
    main()
