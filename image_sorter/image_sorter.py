import os
import logging
import datetime
from PIL import Image

from .face_recognizer import FaceRecognizer
from .utils import (
    copy,
    move,
    mkdir,
    get_file_extension,
    get_file_created_datetime,
)
from .exceptions import NoFacesDetectedError
from .constants import (
    OUTPUT_DIR,
    ARCHIVE_DIR,
    NO_FACES_DETECTED_DIR,
    EXIF_DATETIME_ORIGINAL_TAG,
    EXIF_DATETIME_MODIFIED_TAG,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageSorter:
    def __init__(self):
        self.face_recognizer = FaceRecognizer()

    def load_image(self, image_filepath):
        """Load image using Pillow.open()"""
        return Image.open(image_filepath)

    def get_exif_data(self, image_filepath):
        """Get the EXIF data of the image."""
        try:
            return self.load_image(image_filepath).getexif()
        except Exception:
            logger.exception(
                "Unable to extract the EXIF data from %s.",
                image_filepath,
            )
            return {}

    def get_image_modified_date(self, exif_data):
        """Get the image's modified date, if it exist."""
        image_modified_date = exif_data.get(EXIF_DATETIME_MODIFIED_TAG, None)
        if not image_modified_date:
            return None

        try:
            # truncate the string to remove any (hidden) extra characters
            image_modified_date = image_modified_date[:19]

            # Parse the input string into a datetime object
            dt_obj = datetime.datetime.strptime(
                image_modified_date, "%Y:%m:%d %H:%M:%S"
            )

            # Format the datetime object to YYYYMMDD_HHMMSS
            return dt_obj.strftime("%Y%m%d_%H%M%S")
        except Exception:
            logger.info(
                "Formatting of image_modified_date %s failed.", image_modified_date
            )
            return None

    def get_image_created_date(self, exif_data):
        """Get the image's created date, if it exist."""
        return exif_data.get(EXIF_DATETIME_ORIGINAL_TAG, None)

    def determine_new_image_filename(self, model_name, image_filepath):
        """
        Determines the image's new filename based on the image's model name and
        creation date.
        """
        exif_data = self.get_exif_data(image_filepath)

        image_created_datetime = (
            self.get_image_modified_date(exif_data)
            or self.get_image_created_date(exif_data)
            or get_file_created_datetime(image_filepath)
        )

        new_image_filename = (
            f"{model_name}_{image_created_datetime}{get_file_extension(image_filepath)}"
        )
        return new_image_filename

    def determine_recognized_image_filepath(self, model_name, image_filepath):
        """Determines where the recognized image will be stored."""
        destination_dir = os.path.join(OUTPUT_DIR, model_name)
        mkdir(destination_dir)
        new_image_filename = self.determine_new_image_filename(
            model_name, image_filepath
        )
        # validation/024xc5.jpg  test me
        return os.path.join(destination_dir, new_image_filename)

    def sort_image_by_face_recognition(self, image_filepath):
        """Sorts the image by face recognition."""
        # for recognized_face in self.face_recognizer.recognize_faces(image_filepath):
        #     recognized_image_filepath = self.determine_recognized_image_filepath(
        #         recognized_face, image_filepath
        #     )
        #     copy(image_filepath, recognized_image_filepath)
        #     logger.info("Copied %s to %s", image_filepath, recognized_image_filepath)

        recognized_faces = self.face_recognizer.recognize_faces(image_filepath)
        recognized_image_filepath = self.determine_recognized_image_filepath(
            recognized_faces, image_filepath
        )
        copy(image_filepath, recognized_image_filepath)
        logger.info("Copied %s to %s", image_filepath, recognized_image_filepath)

        # archive processed images
        # TODO zip processed images
        # TODO create separate function
        archive_image_filepath = os.path.join(
            ARCHIVE_DIR, os.path.basename(image_filepath)
        )
        move(image_filepath, archive_image_filepath)

    def sort_images_by_face_recognition(self, image_filepath_list):
        """Sort the images by face recognition."""
        image_processed = 0
        for image_filepath in image_filepath_list:
            try:
                self.sort_image_by_face_recognition(image_filepath)
                image_processed += 1
            except NoFacesDetectedError:
                # TODO create separate function
                no_face_image_filepath = os.path.join(
                    NO_FACES_DETECTED_DIR, os.path.basename(image_filepath)
                )
                move(image_filepath, no_face_image_filepath)

        logger.info("Sorted %s image(s).", image_processed)
