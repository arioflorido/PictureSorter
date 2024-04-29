import argparse
from face_detector import encode_known_faces, recognize_faces
from utils import setup


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--train",
        type=str,
        help="Train a new or an existing model.",
    )
    args = parser.parse_args()
    setup()

    if args.train:
        encode_known_faces(args.train)

    recognize_faces()
