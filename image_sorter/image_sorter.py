import os
import logging
import datetime
from PIL import Image

# from .face_detector import recognize_faces
from .face_recognizer import FaceRecognizer
from .utils import (
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

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)


class ImageSorter:
    def load_image(self, image_filepath):
        """Load image using Pillow.open()"""
        try:
            return Image.open(image_filepath)
        except:
            logger.error(
                "Failed to load image : %s", image_filepath, exc_info=True
            )
            raise

    def get_exif_data(self, image_filepath):
        """Get the EXIF data of the image."""
        try:
            return self.load_image(image_filepath).getexif()
        except Exception:
            logger.error(
                "Unable to extract the EXIF data from %s.",
                image_filepath,
                exc_info=True,
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
            dt_obj = datetime.datetime.strptime(image_modified_date, "%Y:%m:%d %H:%M:%S")

            # Format the datetime object to YYYYMMDD_HHMMSS
            return dt_obj.strftime("%Y%m%d_%H%M%S")
        except Exception:
            logger.error("Formatting of image_modified_date %s failed.", image_modified_date)
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

        new_image_filename = f"{model_name}_{image_created_datetime}{get_file_extension(image_filepath)}"
        return new_image_filename

    def determine_new_image_filepath(self, model_name, image_filepath):
        """Determines the image's new location."""
        output_filepath = os.path.join(OUTPUT_DIR, model_name)
        mkdir(output_filepath)
        new_image_filename = self.determine_new_image_filename(model_name, image_filepath)
        # validation/024xc5.jpg  test me
        return os.path.join(output_filepath, new_image_filename)

    def sort_image_by_face_recognition(self, image_filepath):
        """Sorts the image by face recognition."""
        try:
            face_recognizer = FaceRecognizer()
            for recognized_face in face_recognizer.recognize_faces(image_filepath):
                new_image_filepath = self.determine_new_image_filepath(recognized_face, image_filepath)
                move(image_filepath, new_image_filepath)
                logger.info("Moved %s to %s", image_filepath, new_image_filepath)
        except Exception as error:
            logger.error(error, exc_info=True)
            raise


    def sort_images_by_face_recognition(self, image_filepath_list):
        """Sort the images by face recognition."""
        try:
            for image_filepath in image_filepath_list:
                self.sort_image_by_face_recognition(image_filepath)
        except Exception as error:
            logger.error(error, exc_info=True)
            raise
