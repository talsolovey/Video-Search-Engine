import logging
import yt_dlp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_video(VIDEO_PATH):
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
            return True
    except Exception as e:
        logging.error(f"Failed to download video: {e}")
        return False