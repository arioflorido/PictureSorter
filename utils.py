import os
from pathlib import Path
from constants import REQUIRED_DIRS, ENCODINGS_DIR, VALIDATION_DIR


def setup():
    """Creates required directories if they do not exist."""
    for directory in REQUIRED_DIRS:
        Path(directory).mkdir(exist_ok=True)


def get_face_encodings():
    """Generates the path of the available face encodings."""
    if not os.path.isdir(ENCODINGS_DIR):
        raise ValueError(f"{ENCODINGS_DIR} is not a valid path or directory.")

    for dirpath, _, filenames in os.walk(ENCODINGS_DIR):
        for filename in filenames:
            if filename.lower().endswith(".pkl"):
                yield os.path.join(dirpath, filename)


def get_image_files():
    """Generates the path of the image files from the validation directory"""
    if not os.path.isdir(VALIDATION_DIR):
        raise ValueError(f"{VALIDATION_DIR} is not a valid path or directory.")

    for dirpath, _, filenames in os.walk(VALIDATION_DIR):
        for filename in filenames:
            # TODO Filter image files only
            yield os.path.join(dirpath, filename)
