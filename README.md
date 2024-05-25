# PictureSorter
A Python application that sorts your Pictures based on the faces in it.

## Installation

```bash
pipenv install
```

## Usage
1. Train your models
2. file structure
3. Put your files inside the `validation` directory.
4. `python3 main.py`


## Running tests
You can run test via this command:
```bash
python -m unittest discover tests
```

## Todo
- add documentation - how to use
- add support for 2D / 3D pictures
- add tests
- add filtering - only process image files....
- detect duplicates
- make logging work
- dockerize?
- fix renaming
- fix bug
- show list of available models
- fix logging.... (basicconfig)
- create directory for no face detected.
- detect side-view faces
- improve speed / performance. how? cython?
- detect duplicate files...
- archive processed images (tar.gz)
- improve logging:
    - include counts, etc
- what if multiple faces detected in picture?
    - create model_name_x_model_name.jpg?
- try rotating the image?
- add option to do not use CNN (to avoid memory issue)
  - or choose what option? like --hog-mode or --cnn-mode
- add flag to reprocess no_faces_detected or unknown_faces?>

## Reference
- https://realpython.com/face-recognition-with-python/


algo
  face detection:
    load image
    detect faces
    encode known faces

  face recognition:
    load image
    detect faces
    recognize faces
