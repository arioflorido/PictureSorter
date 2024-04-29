import os
import logging
from face_detector import recognize_faces
from PIL import Image
from utils import (
    get_image_files,
    move,
    mkdir,
    get_file_extension,
    get_file_created_datetime,
)
from constants import OUTPUT_DIR, EXIF_DATETIME_ORIGINAL_TAG, EXIF_DATETIME_MODIFIED_TAG


logger = logging.getLogger(__name__)


def get_image_modified_or_created_date(image_filepath):
    """
    Get the modified date or creation date (whichever is available) of an image
    from the EXIF data. If both dates are not available fro EXIF, get the
    creation date from the file's metadata.
    """
    try:
        with Image.open(image_filepath) as img:
            original_creation_date = None
            exif_data = img.getexif()
            if exif_data:
                original_creation_date = exif_data.get(EXIF_DATETIME_ORIGINAL_TAG, None)

                if original_creation_date:
                    return original_creation_date

                modified_date = exif_data.get(EXIF_DATETIME_MODIFIED_TAG, None)

                if modified_date:
                    return modified_date

            return get_file_created_datetime(image_filepath)
    except Exception:
        return get_file_created_datetime(image_filepath)


def determine_image_filename(image_filepath, model_name):
    """
    Determines the image's new filename based on the image's model name and
    creation date.
    """
    created_datetime = get_image_modified_or_created_date(image_filepath)
    new_image_filename = (
        f"{model_name}_{created_datetime}{get_file_extension(image_filepath)}"
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
