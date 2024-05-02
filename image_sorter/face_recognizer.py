import os
import pickle
from collections import Counter
import logging
from pathlib import Path
import face_recognition

from .constants import ENCODINGS_DIR
from .utils import get_face_encodings

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)


class FaceRecognizer:
    # def __init__(self, image_filepath_list):
    #     self.image_filepath_list = image_filepath_list

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

            if not face_locations:
                logger.info("No faces detected in %s", image_filepath)
                return {}

            return face_recognition.face_encodings(image, face_locations)
        except:
            logger.error(
                "Failed to get the face encodings from the image.", exc_info=True
            )
            raise

    def encode_known_faces(self, model_name, image_filepath_list):
        """
        Detects the face in each image, get its encoding, and groups them
        together in a single dictionary, then save it into a pickle file.
        """
        names = []
        encodings = []
        for image_filepath in image_filepath_list:
            face_encodings = self.get_face_encodings(image_filepath)
            for encoding in face_encodings:
                names.append(model_name)
                encodings.append(encoding)

        encodings = {"names": names, "encodings": encodings}
        self.save_encodings(encodings, model_name)

    def save_encodings(self, encodings, model_name):
        """Saves the encoding to a pickle file."""
        encoding_filename = f"{model_name}.pkl"
        encoding_output = os.path.join(ENCODINGS_DIR, encoding_filename)

        try:
            with open(encoding_output, mode="wb") as fh:
                pickle.dump(encodings, fh)
        except Exception as error:
            logger.error(error, exc_info=True)
            raise

    def recognize_faces(self, image_filepath):
        """
        See if we can recognize the faces in the image using the existing face
        encodings we have in the system.
        """
        recognized_faces = []

        for unknown_face_encodings in self.get_face_encodings(image_filepath):
            recognized_face = self.recognize_face(unknown_face_encodings)
            if recognized_face:
                recognized_faces.append(recognized_face)
        return recognized_faces

    def recognize_face(self, input_face_encodings):
        """
        See if the provided face_encodings matches the existing face_encodings
        we have in the system.
        """
        # TODO: I feel like this needs to be refactored i.e. read pickle before loop or loop then read pickle?
        for (
            existing_face_encodings
        ) in get_face_encodings():  # Refactor get_face_encodings()
            # TODO: def load_encodings()
            with Path(existing_face_encodings).open(mode="rb") as f:
                loaded_encodings = pickle.load(f)

                boolean_matches = face_recognition.compare_faces(
                    loaded_encodings["encodings"], input_face_encodings
                )
                votes = Counter(
                    name
                    for match, name in zip(boolean_matches, loaded_encodings["names"])
                    if match
                )
                if votes:
                    return votes.most_common(1)[0][0]
                return None
