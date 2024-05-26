import os

# class RequiredDirectories:
#     INPUT_DIR = "input"
#     OUTPUT_DIR = "output"
#     TRAINING_DIR = "training"
#     ARCHIVE_DIR = "archive"
#     ENCODINGS_DIR = "encodings"

INPUT_DIR = "input"
OUTPUT_DIR = "output"
TRAINING_DIR = "training"
ARCHIVE_DIR = "archive"
ENCODINGS_DIR = "encodings"
NO_FACES_DETECTED_DIR = os.path.join(OUTPUT_DIR, "no_faces_detected")

REQUIRED_DIRS = (OUTPUT_DIR, TRAINING_DIR, INPUT_DIR, ENCODINGS_DIR, ARCHIVE_DIR)

EXIF_DATETIME_ORIGINAL_TAG = 36867
EXIF_DATETIME_MODIFIED_TAG = 306
UNKNOWN_FACES = "unknown_faces"
NO_FACES_DETECTED = "no_faces_detected"


class FaceDetectionModel:
    HOG = "hog"
    CNN = "cnn"
