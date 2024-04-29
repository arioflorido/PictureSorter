import os
import pickle
from collections import Counter
import logging
from pathlib import Path
import face_recognition
from constants import TRAINING_DIR, ENCODINGS_DIR, UNKNOWN
from utils import get_face_encodings

logger = logging.getLogger(__name__)


def get_training_images(model_name):
    """
    Generates the path of the available images of the specified model name to be
    used in the training.
    """
    training_images_directory = os.path.join(TRAINING_DIR, model_name)

    if not os.path.isdir(training_images_directory):
        raise ValueError(
            f"{training_images_directory} is not a valid path or directory."
        )

    for dirpath, _, filenames in os.walk(training_images_directory):
        for filename in filenames:
            yield os.path.join(dirpath, filename)


def load_image(image_filepath):
    return face_recognition.load_image_file(image_filepath)


def extract_face_locations(image, model="hog"):
    """Detects and returns the locations of faces in the image."""
    return face_recognition.face_locations(image, model=model)


def extract_face_encodings(image_filepath):
    """Extracts the face encodings from the image."""
    # Face encoding is an array of numbers describing the features of the face,
    # and it's used with the main model underlying face recognition to reduce
    # training time while improving the accuracy of a large model. This is known
    # as transfer learning.
    image = load_image(image_filepath)
    face_locations = extract_face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    return (face_locations, face_encodings)


def encode_known_faces(model_name):
    """
    Detects the face in each training image and get its encoding then creates a
    dictionary that puts the names and encodings lists together and denotes
    which list is which. Then, use pickle to save the encodings to disk.
    """
    names = []
    encodings = []
    for image_filepath in get_training_images(model_name):
        _, face_encodings = extract_face_encodings(image_filepath)

        for encoding in face_encodings:
            names.append(model_name)
            encodings.append(encoding)

    name_encodings = {"names": names, "encodings": encodings}
    encoding_filename = f"{model_name}.pkl"

    encoding_output = os.path.join(ENCODINGS_DIR, encoding_filename)
    with open(encoding_output, mode="wb") as fh:
        pickle.dump(name_encodings, fh)


def recognize_faces(image_filepath):
    """Recognize the faces in the images using the available face encodings."""
    input_face_locations, input_face_encodings = extract_face_encodings(image_filepath)

    recognized_faces = []
    unrecognized_faces_coordinates = []
    for bounding_box, unknown_encoding in zip(
        input_face_locations, input_face_encodings
    ):
        recognized_faces.append(_recognize_face(unknown_encoding))
        unrecognized_faces_coordinates.append(bounding_box)

    if unrecognized_faces_coordinates:
        logger.info(
            "Unrecognized %s face(s) detected in '%s'.",
            len(recognized_faces),
            image_filepath,
        )

    return recognized_faces


def _recognize_face(unknown_encoding):
    """Recognize the face in the images using the available face encodings."""
    for face_encodings in get_face_encodings():
        with Path(face_encodings).open(mode="rb") as f:
            loaded_encodings = pickle.load(f)

            boolean_matches = face_recognition.compare_faces(
                loaded_encodings["encodings"], unknown_encoding
            )
            votes = Counter(
                name
                for match, name in zip(boolean_matches, loaded_encodings["names"])
                if match
            )
            if votes:
                return votes.most_common(1)[0][0]
            return UNKNOWN
