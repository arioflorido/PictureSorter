import argparse
from image_sorter.face_detector import encode_known_faces
from image_sorter import ImageSorter
from image_sorter.utils import get_image_files

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
        encode_known_faces(args.train)

    for image_filepath in get_image_files():
        ImageSorter(image_filepath).sort_image_by_face_recognition()
