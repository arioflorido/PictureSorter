import os
import pickle
from collections import Counter
import logging
from pathlib import Path
import face_recognition

from .constants import ENCODINGS_DIR, UNKNOWN
from .utils import get_face_encodings

logger = logging.getLogger(__name__)


class FaceDetector:
    def __init__(self, image_filepath_list, model_name):
        self.model_name = model_name
        self.image_filepath_list = image_filepath_list

    def load_image(self, image_filepath):
        """Load image using face_recognition.load_image_file()"""
        try:
            return face_recognition.load_image_file(image_filepath)
        except:
            logger.error("Failed to load image : %s", image_filepath, exc_info=True)
            raise

    def detect_face_locations(self, image, model="hog"):
        """Detects and returns the locations of faces in the image."""
        try:
            return face_recognition.face_locations(image, model=model)
        except:
            logger.error(
                "Failed to detect face locations from the image.", exc_info=True
            )
            raise

    def get_face_encodings(self, image_filepath):
        """Returns the face encodings from the detected face in the image."""
        try:
            image = self.load_image(image_filepath)
            face_locations = self.detect_face_locations(image)
            return face_recognition.face_encodings(image, face_locations)
        except:
            logger.error(
                "Failed to get the face encodings from the image.", exc_info=True
            )
            raise

    def encode_known_faces(self):
        """
        Detects the face in each training image, get its encoding, and groups
        them together in a single dictionary.
        """
        names = []
        encodings = []
        for image_filepath in self.image_filepath_list:
            face_encodings = self.get_face_encodings(image_filepath)

            for encoding in face_encodings:
                names.append(self.model_name)
                encodings.append(encoding)

        encodings = {"names": names, "encodings": encodings}
        self.save_encodings(encodings)

    def save_encodings(self, encodings):
        """Saves the encoding to a pickle file."""
        encoding_filename = f"{self.model_name}.pkl"
        encoding_output = os.path.join(ENCODINGS_DIR, encoding_filename)

        try:
            with open(encoding_output, mode="wb") as fh:
                pickle.dump(encodings, fh)
        except Exception as error:
            logger.error(error, exc_info=True)
            raise


# def recognize_faces(image_filepath):
#     """Recognize the faces in the images using the available face encodings."""
#     input_face_locations, input_face_encodings = extract_face_encodings(image_filepath)

#     recognized_faces = []
#     unrecognized_faces_coordinates = []
#     for bounding_box, unknown_encoding in zip(
#         input_face_locations, input_face_encodings
#     ):
#         recognized_faces.append(_recognize_face(unknown_encoding))
#         unrecognized_faces_coordinates.append(bounding_box)

#     if unrecognized_faces_coordinates:
#         logger.info(
#             "Unrecognized %s face(s) detected in '%s'.",
#             len(recognized_faces),
#             image_filepath,
#         )

#     return recognized_faces


# def _recognize_face(unknown_encoding):
#     """Recognize the face in the images using the available face encodings."""
#     for face_encodings in get_face_encodings():
#         with Path(face_encodings).open(mode="rb") as f:
#             loaded_encodings = pickle.load(f)

#             boolean_matches = face_recognition.compare_faces(
#                 loaded_encodings["encodings"], unknown_encoding
#             )
#             votes = Counter(
#                 name
#                 for match, name in zip(boolean_matches, loaded_encodings["names"])
#                 if match
#             )
#             if votes:
#                 return votes.most_common(1)[0][0]
#             return UNKNOWN
