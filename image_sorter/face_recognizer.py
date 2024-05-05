import os
import pickle
from collections import Counter
import logging
from pathlib import Path
import face_recognition

from .exceptions import NoFacesDetectedError
from .constants import ENCODINGS_DIR, NO_FACES_DETECTED_DIR, UNKNOWN_FACES
from .utils import get_face_encodings_from_image, mkdir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceRecognizer:

    def load_image(self, image_filepath):
        """Load image using face_recognition.load_image_file()"""
        return face_recognition.load_image_file(image_filepath)

    def determine_no_face_detected_image_filepath(self, image_filepath):
        """Determines where images with no faces detected will be stored."""
        mkdir(NO_FACES_DETECTED_DIR)
        return os.path.join(NO_FACES_DETECTED_DIR, os.path.basename(image_filepath))

    def detect_face_locations_using_hog(self, image):
        """
        Detects and returns the locations of faces in the image using HOG model
        which utilizes the CPU.
        """
        return face_recognition.face_locations(image, model="hog")

    def detect_face_locations_using_cnn(self, image):
        """
        Detects and returns the locations of faces in the image using CNN model
        which utilizes the GPU.
        """
        return face_recognition.face_locations(image, model="cnn")

    def detect_face_locations(self, image):
        """Detects and returns the locations of faces in the image."""
        face_locations = self.detect_face_locations_using_hog(image)
        if not face_locations:
            raise NoFacesDetectedError
        logger.info("Detected %s face(s) in the image.", len(face_locations))
        return face_locations

    def get_face_encodings_from_image(
        self,
        image_filepath,
    ):
        """Returns the face encodings from the detected face in the image."""
        image = self.load_image(image_filepath)
        face_locations = self.detect_face_locations(image)
        return face_recognition.face_encodings(image, face_locations)

    def encode_known_faces(self, model_name, image_filepath_list):
        """
        Detects the face in each image, get its encoding, and groups them
        together in a single dictionary, then save it into a pickle file.
        """
        names = []
        encodings = []
        for image_filepath in image_filepath_list:
            try:
                face_encodings = self.get_face_encodings_from_image(image_filepath)
            except NoFacesDetectedError:
                logger.error("No faces detected in %s.", image_filepath)
                raise
            for face_encoding in face_encodings:
                names.append(model_name)
                encodings.append(face_encoding)

        extracted_face_encodings = {"names": names, "encodings": encodings}
        self.save_encodings_to_pickle(extracted_face_encodings, model_name)

    def save_encodings_to_pickle(self, encodings, model_name):
        """Saves the encoding to a pickle file."""
        encoding_filename = f"{model_name}.pkl"
        encoding_output = os.path.join(ENCODINGS_DIR, encoding_filename)

        with open(encoding_output, mode="wb") as fh:
            pickle.dump(encodings, fh)

    def recognize_faces(self, image_filepath):
        """
        See if we can recognize the faces in the image using the existing face
        encodings we have in the system.
        """
        recognized_faces = set()

        for face_encodings in self.get_face_encodings_from_image(image_filepath):
            if len(face_encodings) > 0:
                recognized_faces.add(self.recognize_face(face_encodings))

        if recognized_faces:
            return "_".join(sorted(recognized_faces))
        return None
        # return recognized_faces

    def recognize_face(self, input_face_encodings):
        """
        See if the provided face_encodings matches the existing face_encodings
        we have in the system.
        """
        # TODO: I feel like this needs to be refactored i.e. read pickle before loop or loop then read pickle?
        # Refactor get_face_encodings_from_image()
        for existing_face_encodings in get_face_encodings_from_image():
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
                return UNKNOWN_FACES
