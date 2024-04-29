import os
from pathlib import Path
from datetime import datetime
from constants import (
    REQUIRED_DIRS,
    ENCODINGS_DIR,
    VALIDATION_DIR,
    EXIF_DATETIME_ORIGINAL_TAG,
)


def get_file_extension(filepath):
    """Returns the file extension."""
    return os.path.splitext(filepath)[1]


def move(old_filepath, new_filepath):
    """Moves a file to a new location."""
    os.rename(old_filepath, new_filepath)


def mkdir(directory_name):
    """Creates a directory."""
    Path(directory_name).mkdir(exist_ok=True)


def setup():
    """Creates required directories if they do not exist."""
    for directory in REQUIRED_DIRS:
        mkdir(directory)


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
            print(filename)
            # TODO Filter image files only
            yield os.path.join(dirpath, filename)


def get_file_created_datetime(filepath):
    """Returns the created time of the file."""
    return datetime.fromtimestamp(os.path.getctime(filepath)).strftime("%Y%m%d_%H%M%S")
