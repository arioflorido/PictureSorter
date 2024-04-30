import argparse
from image_sorter.face_detector import encode_known_faces
from image_sorter.image_sorter import sort_image_by_face_recognition
from image_sorter.utils import setup
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-tm",
        "--train-model",
        type=str,
        action="store",
        help="Train a new or an existing model.",
    )
    # parser.add_argument(
    #     "-t",
    #     "--test",
    #     action="store_true",
    #     help="Run tests.",
    # )
    args = parser.parse_args()
    setup()

    # if args.test:
    # print(1)
    # import unittest
    # from tests import test_utils
    # unittest.main(module=test_utils)
    #     sys.exit()
    if args.train_model:
        encode_known_faces(args.train)

    # sort_image_by_face_recognition()

    # import unittest
    # from tests import test_utils
    # unittest.main(module=test_utils)
