import os
import yt_dlp
import cv2
from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector
import moondream as md
from PIL import Image
import json

# Initialize the Moondream2 model
model = md.vl(model="./moondream-2b-int8.mf")

# Paths
output_path = 'scene_frames'
captions_file = 'scene_captions.json'
video_path = 'downloaded_video.mp4'


def main():
    # If captions file exists, skip download, creation of image, and captioning
    if not os.path.exists(captions_file):
        # Ensure output directory for scene frames exists
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Search for a "Super Mario movie trailer" on YouTube and download it.
        ydl_opts = {
            'format': 'bestvideo',
            'noplaylist': True,
            'outtmpl': 'downloaded_video.mp4'
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download(['ytsearch:super mario movie trailer'])
        except Exception as e:
            print(f"Failed to download video: {e}")
            return

        # Set up pyscenedetect to detect scenes
        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector())

        try:
            video_manager.set_downscale_factor()
            video_manager.start()
            scene_manager.detect_scenes(frame_source=video_manager)
        finally:
            video_manager.release()

        scene_list = scene_manager.get_scene_list()

        # Function to save a frame at the specified number
        def save_frame(frame_number, output_path):
            video_cap = cv2.VideoCapture(video_path)
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, image = video_cap.read()
            frame_path = None
            if success:
                frame_path = f"{output_path}/scene_{frame_number}.png"
                cv2.imwrite(frame_path, image)
            else:
                print(f"Failed to capture frame at {frame_number}")
            video_cap.release()
            return frame_path

        # Save the frame at the start of each scene
        for i, scene in enumerate(scene_list):
            start_frame = scene[0].frame_num  # Start frame of the scene
            save_frame(start_frame, output_path)

        # Dictionary to hold scene captions
        scene_captions = {}

        # Save the frame at the start of each scene and generate captions
        for i, scene in enumerate(scene_list):
            start_frame = scene[0].frame_num  # Start frame of the scene
            frame_path = save_frame(start_frame, output_path)
            if frame_path:
                # Load the image using PIL
                image = Image.open(frame_path)
                # Generate caption using Moondream2
                caption = model.caption(image)["caption"]
                # Store the caption with the scene number
                scene_captions[f"scene_{start_frame}"] = caption

        # Save the scene captions to a JSON file
        with open(captions_file, 'w') as json_file:
            json.dump(scene_captions, json_file, indent=4)

    # Functionality to search scenes by a word
    if os.path.exists(captions_file):
        with open(captions_file, 'r') as json_file:
            scene_captions = json.load(json_file)

        print("Search the video using a word:")
        search_word = input().strip().lower()

        matching_scenes = [scene for scene, caption in scene_captions.items()
                           if search_word in caption.lower()]

        if matching_scenes:
            print("Scenes containing the word:\n")
            scene_images = []
            for scene in matching_scenes:
                print(f"{scene}: {scene_captions[scene]}")
                image_path = f"{output_path}/{scene}.png"
                if os.path.exists(image_path):
                    try:
                        scene_images.append(Image.open(image_path))
                    except Exception as e:
                        print(f"Error loading image {image_path}: {e}")

            if scene_images:
                # Create a collage of all matching scene images
                try:
                    widths, heights = zip(*(img.size for img in scene_images))
                    total_width = sum(widths)
                    max_height = max(heights)

                    collage = Image.new('RGB', (total_width, max_height))

                    x_offset = 0
                    for img in scene_images:
                        collage.paste(img, (x_offset, 0))
                        x_offset += img.size[0]

                    # Save and display the collage
                    collage.save('collage.png')
                    print("Collage saved as 'collage.png'")
                    collage.show()
                except Exception as e:
                    print(f"Error creating collage: {e}")
            else:
                print("No images available for matching scenes.")
        else:
            print("No scenes found containing the word.")


if __name__ == "__main__":
    main()
