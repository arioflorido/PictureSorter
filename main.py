import argparse

# from image_sorter.face_detector import encode_known_faces
from image_sorter import ImageSorter
from image_sorter import FaceRecognizer
from image_sorter.utils import get_image_files, get_training_images

from image_sorter.utils import setup

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--train",
        type=str,
        action="store",
        help="Train a new or an existing model.",
    )
    args = parser.parse_args()
    setup()

    if args.train:
        model_name = args.train
        training_images = get_training_images(model_name)
        face_recognizer = FaceRecognizer()
        face_recognizer.encode_known_faces(model_name, training_images)

    images_for_sorting = get_image_files()
    image_sorter = ImageSorter()
    image_sorter.sort_images_by_face_recognition(images_for_sorting)
