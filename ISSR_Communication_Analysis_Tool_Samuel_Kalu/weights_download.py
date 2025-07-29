import gdown
import os

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

print(" Download complete.")