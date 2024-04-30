import os
import logging
from PIL import Image
from .face_detector import recognize_faces
from .utils import (
    get_image_files,
    move,
    mkdir,
    get_file_extension,
    get_file_created_datetime,
)
from .constants import (
    OUTPUT_DIR,
    EXIF_DATETIME_ORIGINAL_TAG,
    EXIF_DATETIME_MODIFIED_TAG,
)


logger = logging.getLogger(__name__)


def get_exif_data(image_filepath):
    """Get the EXIF data of an image."""
    try:
        with Image.open(image_filepath) as img:
            return img.getexif()
    except Exception:
        return {}


def get_image_modified_date(image_filepath):
    """Get the image's modified date, if it exist."""
    return get_exif_data(image_filepath).get(EXIF_DATETIME_MODIFIED_TAG, None)


def get_image_created_date(image_filepath):
    """Get the image's created date, if it exist."""
    return get_exif_data(image_filepath).get(EXIF_DATETIME_ORIGINAL_TAG, None)


def determine_image_filename(image_filepath, model_name):
    """
    Determines the image's new filename based on the image's model name and
    creation date.
    """
    image_created_datetime = (
        get_image_modified_date(image_filepath)
        or get_image_created_date(image_filepath)
        or get_file_created_datetime(image_filepath)
    )

    new_image_filename = (
        f"{model_name}_{image_created_datetime}{get_file_extension(image_filepath)}"
    )
    return new_image_filename


def sort_image_by_face_recognition():
    """Sorts the images by face recognition"""
    for image_filepath in get_image_files():
        try:
            for recognized_face in recognize_faces(image_filepath):
                output_filepath = os.path.join(OUTPUT_DIR, recognized_face)
                mkdir(output_filepath)
                new_image_filename = determine_image_filename(
                    image_filepath, recognized_face
                )
                new_image_filepath = os.path.join(output_filepath, new_image_filename)

                move(image_filepath, new_image_filepath)
                logger.info(new_image_filepath)
        except Exception as error:
            logger.error(error)
            raise
