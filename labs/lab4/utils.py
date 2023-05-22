from glob import glob
import cv2
import numpy as np
import dlib
import os

face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(
    "models/shape_predictor_68_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1(
    "models/dlib_face_recognition_resnet_model_v1.dat")

VALID_ECTENSIONS = ['.png', '.jpg', '.jpeg']


def get_image_paths(root_dir, class_names):
    image_paths = []

    for class_name in class_names:
        class_dir = os.path.sep.join([root_dir, class_name])
        class_file_paths = glob(os.path.sep.join([class_dir, '*.*']))

        for file_path in class_file_paths:
            ext = os.path.splitext(file_path)[1]

            if ext.lower() not in VALID_ECTENSIONS:
                print("Skipping file: {}".format(file_path))
                continue

            image_paths.append(file_path)

    return image_paths


def face_rects(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = face_detector(gray, 1)
    # return the bounding boxes
    return rects


def face_landmarks(image):
    return [shape_predictor(image, face_rect)
            for face_rect in face_rects(image)
            ]

def face_encodings(image):
    # compute the facial embeddings for each face
    # in the input image. the compute_face_descriptor
    # function returns a 128-d vector that describes the face in an image
    return [
        np.array(face_encoder.compute_face_descriptor(image, face_landmark))
        for face_landmark in face_landmarks (image)
    ]

def nb_of_matches (known_encodings, unknown_encoding) :
    # compute the Euclidean distance between the current face encoding
    # and all the face encodings in the database
    distances = np. linalg.norm(known_encodings - unknown_encoding, axis=1)
    # keep only the distances that are less than the threshold
    small_distances = distances <= 0.6
    # return the number of matches
    return sum(small_distances)

