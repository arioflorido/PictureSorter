import os
import logging
from face_detector import recognize_faces
from utils import get_filename, get_image_files, move, mkdir
from constants import OUTPUT_DIR

logger = logging.getLogger(__name__)

# output
# recognized_faces > mkdir > if less than one recognized face, move, else,
# copy until recognized face is exhausted.


def sort_image_by_face_recognition():
    """Sorts images by face recognition"""
    for image_filepath in get_image_files():
        for recognized_face in recognize_faces(image_filepath):
            output_filepath = os.path.join(OUTPUT_DIR, recognized_face)
            mkdir(output_filepath)
            new_image_filepath = os.path.join(
                output_filepath, get_filename(image_filepath)
            )
            move(image_filepath, new_image_filepath)
            logger.info(new_image_filepath)
