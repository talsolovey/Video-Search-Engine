import os
import yt_dlp
import cv2
from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector

# Ensure output directory for scene frames exists
output_path = 'scene_frames'
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
    exit(1)


# Set up pyscenedetect to detect scenes
video_path = 'downloaded_video.mp4'
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
    if success:
        cv2.imwrite(f"{output_path}/scene_{frame_number}.png", image)
    else:
        print(f"Failed to capture frame at {frame_number}")
    video_cap.release()

# Save the frame at the start of each scene
for i, scene in enumerate(scene_list):
    start_frame = scene[0].frame_num  # Start frame of the scene
    save_frame(start_frame, output_path)