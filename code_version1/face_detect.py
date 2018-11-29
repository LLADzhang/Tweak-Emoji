import numpy as np
import cv2
import os
import shutil

if __name__ == '__main__':
    data_dir = '../data/train'
    face_dir = '../face_data/train'

    face_cascade = cv2.CascadeClassifier('../classifier/haarcascade_frontalface_default.xml')

    for directory in os.listdir(data_dir):
        if not os.path.exists(os.path.join(face_dir, directory)):
            os.mkdir(os.path.join(face_dir, directory))
        for img in os.listdir(os.path.join(data_dir, directory)):
            orig = cv2.imread(os.path.join(data_dir, directory, img), 0)
            faces = face_cascade.detectMultiScale(orig, 1.3, 5)
            (x, y, w, h) = faces[0]
            face = orig[x:x+w][y:y+h]
            cv2.imwrite(os.path.join(face_dir, directory, img), face)
