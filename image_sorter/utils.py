import os
from pathlib import Path
from datetime import datetime
from .constants import REQUIRED_DIRS, ENCODINGS_DIR, VALIDATION_DIR, TRAINING_DIR


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


def get_training_images(model_name):
    """
    Generates the path of the available images of the specified model name to be
    used for training.
    """
    training_images_directory = os.path.join(TRAINING_DIR, model_name)

    if not os.path.exists(training_images_directory):
        raise ValueError(f"{training_images_directory} does not exists.")
    if not os.path.isdir(training_images_directory):
        raise ValueError(f"{training_images_directory} is not a  directory.")

    training_images = []
    for dirpath, _, filenames in os.walk(training_images_directory):
        for filename in filenames:
            training_images.append(os.path.join(dirpath, filename))
    return training_images
