import argparse

# from image_sorter.face_detector import encode_known_faces
from image_sorter import ImageSorter
from image_sorter import FaceRecognizer
from image_sorter.utils import get_image_files, get_training_images
from image_sorter.utils import setup
from image_sorter.constants import FaceDetectionModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--train",
        type=str,
        action="store",
        help="Train a new or an existing model.",
    )
    # TODO modify to select between hog and CNN
    # parser.add_argument(
    #     "-fdm",
    #     "--face-detection-model",
    #     type=str,
    #     default="hog",
    #     action="store",
    #     help="Choose what face detection model to use. (Uses HOG model by default).",
    # )
    parser.add_argument(
        "-fdm",
        "--face-detection-model",
        choices=[FaceDetectionModel.HOG, FaceDetectionModel.CNN],
        default=FaceDetectionModel.HOG,
        help="Choose what face detection model to use."
    )

    args = parser.parse_args()
    setup()

    if args.train:
        model_name = args.train
        training_images = get_training_images(model_name)
        face_recognizer = FaceRecognizer()

        # Encode face separately using HOG and CNN face detection models
        face_recognizer.encode_known_faces(model_name, training_images, FaceDetectionModel.HOG)
        face_recognizer.encode_known_faces(model_name, training_images, FaceDetectionModel.CNN)
    else:
        images_for_sorting = get_image_files()
        image_sorter = ImageSorter()
        image_sorter.sort_images_by_face_recognition(images_for_sorting, args.face_detection_model)
