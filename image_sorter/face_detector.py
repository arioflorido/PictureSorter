# import os
# import pickle
# from collections import Counter
# import logging
# from pathlib import Path
# import face_recognition

# from .constants import ENCODINGS_DIR, UNKNOWN
# from .utils import get_face_encodings

# logger = logging.getLogger(__name__)


# class FaceDetector:
#     def __init__(self, image_filepath_list):
#         self.image_filepath_list = image_filepath_list

#     def load_image(self, image_filepath):
#         """Load image using face_recognition.load_image_file()"""
#         try:
#             return face_recognition.load_image_file(image_filepath)
#         except:
#             logger.error("Failed to load image : %s", image_filepath, exc_info=True)
#             raise

#     def detect_face_locations(self, image, model="hog"):
#         """Detects and returns the locations of faces in the image."""
#         try:
#             return face_recognition.face_locations(image, model=model)
#         except:
#             logger.error(
#                 "Failed to detect face locations from the image.", exc_info=True
#             )
#             raise

#     def get_face_encodings(self, image_filepath):
#         """Returns the face encodings from the detected face in the image."""
#         try:
#             image = self.load_image(image_filepath)
#             face_locations = self.detect_face_locations(image)
#             return face_recognition.face_encodings(image, face_locations)
#         except:
#             logger.error(
#                 "Failed to get the face encodings from the image.", exc_info=True
#             )
#             raise

#     def encode_known_faces(self, model_name):
#         """
#         Detects the face in each image, get its encoding, and groups them
#         together in a single dictionary, then save it into a pickle file.
#         """
#         names = []
#         encodings = []
#         for image_filepath in self.image_filepath_list:
#             face_encodings = self.get_face_encodings(image_filepath)

#             for encoding in face_encodings:
#                 names.append(model_name)
#                 encodings.append(encoding)

#         encodings = {"names": names, "encodings": encodings}
#         self.save_encodings(encodings, model_name)

#     def save_encodings(self, encodings, model_name):
#         """Saves the encoding to a pickle file."""
#         encoding_filename = f"{model_name}.pkl"
#         encoding_output = os.path.join(ENCODINGS_DIR, encoding_filename)

#         try:
#             with open(encoding_output, mode="wb") as fh:
#                 pickle.dump(encodings, fh)
#         except Exception as error:
#             logger.error(error, exc_info=True)
#             raise
