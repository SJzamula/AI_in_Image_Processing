import os
import pickle
import cv2
from utils import face_encodings, get_image_paths

root_dir = "dataset"
class_names = os.listdir(root_dir)

#get the paths to the images
image_paths = get_image_paths(root_dir, class_names)
#initialize a dictionary to store the name
#of each person and the corresponding encodings
name_encoding_dict = {}

#initialize the number of images processed
nb_current_image = 1
#now we can loop over the image paths, locate the faces, and encode them
for image_path in image_paths:
    print(f"Image processed {nb_current_image}/{len(image_paths)}")
    # load the image
    image = cv2.imread(image_path)
    # get the face embeddings
    encodings = face_encodings(image)
    # get the name from the image path
    name = image_path.split(os.path.sep)[-2]
    # get the encodings for the current name
    e = name_encoding_dict.get(name, [])
    # update the list of encodings for the current name
    e.extend(encodings)
    # update the list of encodings for the current name
    name_encoding_dict[name] = e
    nb_current_image += 1

#save the name encodinga dictionary to disk
with open("encoding.pickle", "wb") as f:
    pickle.dump(name_encoding_dict, f)
    