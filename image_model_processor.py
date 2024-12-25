# video_processing.py
import os
import cv2
import moondream as md
from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector
from PIL import Image
import json
import logging
from rapidfuzz import fuzz
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the Moondream2 model
def initialize_moondream_model():
    logging.info("Initializing Moondream2 model...")
    try:
        model = md.vl(model="./moondream-2b-int8.mf")
    except Exception as e:
        logging.error(f"Failed to initialize Moondream2 model: {e}. Exiting.")
        return
    logging.info("Moondream2 model initialized.")
    return model

def process_with_image_model(model, captions_file, scene_dir, collage_path, video_path):
    """
    Process the video frames using the image model.
    """
    # If captions file exists, skip download, creation of image, and captioning
    if not os.path.exists(captions_file):
        # Ensure output directory for scene frames exists
        if not os.path.exists(scene_dir):
            os.makedirs(scene_dir)
            logging.info("Created output directory for scene frames.")
        
            # Detect scenes and save frames
            scene_list = detect_and_save_scene_frames(video_path, scene_dir)
            if not scene_list:
                logging.error("Failed to detect scenes. Exiting.")   
                return

        # Generate captions for the scenes
        scene_captions = generate_captions(scene_dir, captions_file, model)
        
    # Load scene captions from file
    scene_captions = load_captions(captions_file)
    if not scene_captions:
        logging.error("Failed to load scene captions. Exiting.")
        return
    matching_scenes = search_with_autocomplete(scene_captions)
    if not matching_scenes:
        logging.warning("No matching scenes found. Exiting.")
        return
    create_collage(matching_scenes, scene_dir, collage_path)
    if not os.path.exists('collage.png'):
        logging.warning("No collage created. Exiting.")
        return

def detect_and_save_scene_frames(video_path, output_path):
    """
    Detects scenes in the video and saves the frames at the start of each scene.
    """
    # Set up pyscenedetect to detect scenes
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())

    try:
        video_manager.set_downscale_factor()
        video_manager.start()
        logging.info("Detecting scenes...")
        scene_manager.detect_scenes(frame_source=video_manager)
    finally:
        video_manager.release()

    scene_list = scene_manager.get_scene_list()
    logging.info(f"Detected {len(scene_list)} scenes.")

    # Function to save a frame at the specified number
    def save_frame(scene_index, frame_number, output_path):
        video_cap = cv2.VideoCapture(video_path)
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, image = video_cap.read()
        frame_path = None
        if success:
            frame_path = f"{output_path}/scene_{scene_index + 1}.png"
            cv2.imwrite(frame_path, image)
            logging.info(f"Saved frame {frame_number} as {frame_path}.")
        else:
            logging.warning(f"Failed to capture frame at {frame_number}.")
        video_cap.release()
        return frame_path

    # Save the frame at the start of each scene
    for i, scene in enumerate(scene_list):
        start_frame = scene[0].frame_num  # Start frame of the scene
        save_frame(i, start_frame, output_path)

    return scene_list

def generate_captions(scene_dir, captions_file, model):
    """
    Generates captions for the scene images using the image model.
    """
    # Dictionary to hold scene captions
    scene_captions = {}

    scene_files = os.listdir(scene_dir)
    scene_files = [f for f in scene_files if f.endswith('.png')]

    logging.info("Generating captions for each scene...")
    for scene_file in scene_files:
        scene_num = int(scene_file.split('_')[1].split('.')[0])
        image_path = os.path.join(scene_dir, scene_file)
        try:
            image = Image.open(image_path)
            caption = model.caption(image)["caption"] 
            scene_captions[f"scene_{scene_num}"] = caption
            logging.info(f"Scene {scene_num}: {caption}")
        except Exception as e:
            logging.error(f"Error captioning {image_path}: {e}")
            scene_captions[f"scene_{scene_num}"] = ""

    with open(captions_file, 'w', encoding='utf-8') as f:
        json.dump(scene_captions, f, ensure_ascii=False, indent=4)
    logging.info(f"Captions saved to '{captions_file}'.")
    return scene_captions

def load_captions(captions_file):
    """
    Loads captions from the JSON file.
    """
    if not os.path.exists(captions_file):
        logging.warning(f"Captions file '{captions_file}' does not exist.")
        return {}
    with open(captions_file, 'r', encoding='utf-8') as f:
        captions = json.load(f)
    logging.info(f"Loaded captions from '{captions_file}'.")
    return captions


def search_with_autocomplete(scene_captions):
    """
    Searches the video scenes based on user input with auto-completion.
    """
    # Extract unique words from captions for auto-completion
    unique_words = set()
    for caption in scene_captions.values():
        unique_words.update(caption.lower().split())

    completer = WordCompleter(list(unique_words), ignore_case=True)

    print("Search the video using a word:")
    search_word = prompt("Enter search term: ",
                         completer=completer).strip().lower()

    matching_scenes = []
    for scene, caption in scene_captions.items():
        similarity = fuzz.partial_ratio(search_word, caption.lower())
        if similarity > 70:
            matching_scenes.append(scene)
            
    logging.info(f"Found {len(matching_scenes)} matching scenes for the term '{search_word}'.")
    return matching_scenes

def create_collage(scene_images, scene_dir, collage_file):
    """
    Creates a collage of scene images.
    """
    # Create a collage of all matching scene images
    if not scene_images:
        logging.warning("No matching scenes to create collage.")
        return
    
    images = []
    for scene in scene_images:
        image_path = os.path.join(scene_dir, f"{scene}.png")
        if os.path.exists(image_path):
            try:
                img = Image.open(image_path)
                images.append(img)
            except Exception as e:
                logging.error(f"Error loading image '{image_path}': {e}")

    if not images:
        logging.warning("No images available to create a collage.")
        return

    # Define collage parameters
    cols = min(5, len(images))
    rows = (len(images) + cols - 1) // cols
    thumb_width = 200
    thumb_height = 112  # Assuming 16:9 aspect ratio

    collage_width = cols * thumb_width
    collage_height = rows * thumb_height

    collage_image = Image.new('RGB', (collage_width, collage_height), color=(255, 255, 255))

    for idx, img in enumerate(images):
        img = img.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        x = (idx % cols) * thumb_width
        y = (idx // cols) * thumb_height
        collage_image.paste(img, (x, y))

    collage_image.save(collage_file)
    collage_image.show()
    logging.info(f"Collage saved as '{collage_file}'.")