import gdown
import os
from loguru import logger


logger.info("Starting the download of model weights...This would take a while")
# === CONFIG ===
FOLDER_URL = "https://drive.google.com/drive/folders/14psZGt00sC8J08aAPjIzzeaW4dpRhNOY"
DEST_DIR = os.path.join(os.getcwd(), 'model_weights')

# Download the folder
gdown.download_folder(
    url=FOLDER_URL,
    output=DEST_DIR,
    quiet=False,
    use_cookies=False,
    remaining_ok=True
)

logger.success(" Download complete.")