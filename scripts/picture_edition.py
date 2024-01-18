import os, numpy as np, cv2 as cv
from PIL import Image, ImageOps
from IPython.display import display

import numpy as np
from deepface import DeepFace
from retinaface import RetinaFace
from tqdm import tqdm
from matplotlib.pyplot import cm
import cv2 as cv
import os


#### Functions

def clean_pictures(from_dir, to_dir = None, rename = True, rotate = True):
    if to_dir is None:
        if "clean" not in os.listdir(from_dir):
            os.mkdir(from_dir + "clean/")
        to_dir = from_dir + "clean/"
    
    for img_id, img_name in enumerate(next(os.walk(from_dir))[2]): 
        if "ref" in img_name:
            save_name = "ref.jpg"
        else:
            save_name = ".".join([str(img_id+1), img_name.split(".")[-1]]) if rename else img_name
        with Image.open(from_dir + img_name) as im:
            img = ImageOps.exif_transpose(im) if rotate else im
            img.save(to_dir + save_name)

def rectangle_to_square(x1, y1, x2, y2):
    """Transforms rectangle's opposite vertices positions into those of the circumscribed square"""
    w, h = x2-x1, y2-y1
    if h>w:
        x1, x2 = (x1+x2-h)//2, (x1+x2+h)//2
    else:
        y1, y2 = (y1+y2-w)//2, (y1+y2+w)//2
    return x1, y1, x2, y2

def crop_resize(img_dir, face, mode = "cropped", square = True, margin = 20, resolution = 512): 

    #Open img and get its size
    with Image.open(img_dir) as img:
        W, H = img.size
        
        faces = {"face_1" : face} if "facial_area" in face else face
        result_imgs = {"face_1" : img}
        rectangle_colors = list(cm.plasma(np.linspace(0.3, 1, len(faces))))
        for face_id, face in faces.items():
            
            #Get the box around the face
            x1, y1, x2, y2 = face["facial_area"]

            if square:
                x1, y1, x2, y2 = rectangle_to_square(x1, y1, x2, y2)

            margin_divisor = int(100/margin)
            w, h = x2-x1, y2-y1
            x1, x2 = x1-w//margin_divisor, x2 + w//margin_divisor
            y1, y2 = y1-h//margin_divisor, y2 + h//margin_divisor
            
            if mode=="cropped":
                try:
                    cropped_img = img.resize((resolution, resolution), box = (x1, y1, x2, y2))
                except:
                    cropped_img = img.resize((resolution, resolution), box = tuple(face["facial_area"]))
                result_imgs[face_id] = cropped_img
            elif mode=="full":
                rectangle_color = rectangle_colors[int(face_id.replace("face_", ""))-1][:3]*255#(255, 0, 0)
                line_width = max(W, H)//500

                img_with_rectangle = cv.rectangle(np.array(result_imgs["face_1"]), (x1, y1), (x2, y2), rectangle_color, line_width)
                img_with_rectangle = Image.fromarray(img_with_rectangle)
                result_imgs["face_1"] = img_with_rectangle
            else:
                print("Wrong mode")
    return result_imgs["face_1"] if len(result_imgs)==1 else result_imgs
    
def check_and_crop_faces(img_path, ref_name = "ref.jpg"):
    """Automatically goes through all images and crops around the face found in the 'ref'"""

    #Create dedicated folder if it does not exist
    if "cropped" not in os.listdir(img_path):
        os.mkdir(img_path + "cropped/")

    clean_path = img_path + "clean/"
    img_files = next(os.walk(clean_path))[2]
    missing_faces = {}
    for img_name in tqdm(img_files):
        img_dir = clean_path + img_name
        faces = RetinaFace.detect_faces(img_dir)
        correct_faces = {}
        if type(faces)==dict: #Deal with weird RetinaFaces types, which is hard to understand because of lack of documentation...
            if len(faces)>=2:
                for face_id in faces:
                    img = crop_resize(img_dir, face = faces[face_id])
                    face_img_dir = img_dir.replace(".jpg", "") + f"_{face_id}.jpg"
                    img.save(face_img_dir)
                    looks_like_ref = DeepFace.verify(img1_path = face_img_dir, img2_path = clean_path + ref_name, model_name = "ArcFace", detector_backend = "retinaface", enforce_detection = False, prog_bar = False)
                    os.remove(face_img_dir)
                    if looks_like_ref["verified"]:
                        correct_faces[face_id] = faces[face_id]
            else:
                correct_faces["face_1"] = faces["face_1"]
            
            if len(correct_faces)==0 :
                print(f"No matching face could be found for {img_name}")
                missing_faces[img_name] = faces
                display(crop_resize(img_dir, face = faces, mode = "full", margin = 15))
            elif len(correct_faces)>=2:
                print(f"Warning: multiple matching faces were found for {img_name}")
            for _, face in correct_faces.items():
                img = crop_resize(img_dir, face = face, margin = 15).save(img_path + "cropped/" + img_name)
        else:
            print(f"Nothing could be found in {img_name}")
            display(Image.open(img_dir))

        
    return missing_faces

def display_face_detection(img_path, img_name):
    """Show the results of face detection on img"""
    img_dir = img_path + img_name
    faces = RetinaFace.detect_faces(img_dir)
    img = crop_resize(img_dir, face = faces, mode = "full", margin = 15)
    return img

def missing_faces_manual_adjustment(missing_faces, img_name_to_face, img_path):
    """Crop around target face when automatic way didn't work"""
    for img_name, faces in tqdm(missing_faces.items()):
        face = missing_faces[img_name][img_name_to_face[img_name]]
        crop_resize(img_path + "clean/" + img_name, face = face, margin = 15).save(img_path + "cropped/" + img_name)


def data_augment(img_path):
    """Simply flip images to have more samples (actually already implemented in DreamBooth/LoRA training)"""
    img_files = next(os.walk(img_path))[2]
    for img_name in img_files:
        with Image.open(img_path + img_name) as im:
            im.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT).save(img_path + "flipped_" + img_name)
            
## Functions for Insta use case
            
def square_picture(img_path, img_name):
    "Make pictures square for Insta"
    img = cv.imread(img_path + img_name)
    H, W = img.shape[:2]
    if W>H:
        rectangle_height = (W-H)//2
        wide_rectangle = np.zeros((rectangle_height, W,3), np.uint8)
        res = cv.vconcat([wide_rectangle, img, wide_rectangle])
        cv.imwrite(img_path + "squared/" + img_name, res)
    else:
        rectangle_width = (H-W)//2
        tall_rectangle = np.zeros((H, rectangle_width,3), np.uint8)
        res = cv.hconcat([tall_rectangle, img, tall_rectangle])
        cv.imwrite(img_path + "squared/" + img_name, res)

def save_square_pictures(img_path):
    """Square all pictures in folder and save them in a dedicated '/square' folder"""

    #Create dedicated folder if it does not exist
    if "squared" not in os.listdir(img_path):
        os.mkdir(img_path + "squared/")

    #Square pictures and save them in folder
    for img_name in list(next(os.walk(img_path))[2]):
        if img_name not in list(next(os.walk(img_path + "squared/"))[2]):
            square_picture(img_path, img_name)