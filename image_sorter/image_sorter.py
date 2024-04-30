import os
import logging
from PIL import Image
# from .face_detector import recognize_faces
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


logger = logging.getLogger(__name__)


class ImageSorter:
    def __init__(self, image_filepath):
        self.image_filepath = image_filepath
        self.exif_data = self.get_exif_data()

    def load_image(self):
        """Load image using Pillow.open()"""
        try:
            return Image.open(self.image_filepath)
        except:
            logger.error(
                "Failed to load image : %s", self.image_filepath, exc_info=True
            )
            raise

    def get_exif_data(self):
        """Get the EXIF data of the image."""
        try:
            return self.load_image().getexif()
        except:
            logger.error(
                "Unable to extract the EXIF data from %s.",
                self.image_filepath,
                exc_info=True,
            )
            return {}

    def get_image_modified_date(self):
        """Get the image's modified date, if it exist."""
        return self.exif_data.get(EXIF_DATETIME_MODIFIED_TAG, None)

    def get_image_created_date(self):
        """Get the image's created date, if it exist."""
        return self.exif_data.get(EXIF_DATETIME_ORIGINAL_TAG, None)

    def determine_new_image_filename(self, model_name):
        """
        Determines the image's new filename based on the image's model name and
        creation date.
        """
        image_created_datetime = (
            self.get_image_modified_date()
            or self.get_image_created_date()
            or get_file_created_datetime(self.image_filepath)
        )

        new_image_filename = f"{model_name}_{image_created_datetime}{get_file_extension(self.image_filepath)}"
        return new_image_filename

    # def sort_image_by_face_recognition(self):
    #     """Sorts the images by face recognition."""
    #     try:
    #         for recognized_face in recognize_faces(self.image_filepath):
    #             output_filepath = os.path.join(OUTPUT_DIR, recognized_face)
    #             mkdir(output_filepath)
    #             new_image_filename = self.determine_new_image_filename(recognized_face)
    #             new_image_filepath = os.path.join(output_filepath, new_image_filename)

    #             move(self.image_filepath, new_image_filepath)
    #             logger.info(new_image_filepath)
    #     except Exception as error:
    #         logger.error(error)
    #         raise
