import numpy as np
import cv2
import os
import shutil

if __name__ == '__main__':
    data_dir = '../data/train'
    face_dir = '../face_data/train'

    face_cascade = cv2.CascadeClassifier('../classifier/haarcascade_frontalface_default.xml')

    for directory in os.listdir(data_dir):
        if directory[0] != '.':
            if not os.path.exists(os.path.join(face_dir, directory)):
                os.mkdir(os.path.join(face_dir, directory))
            for img in os.listdir(os.path.join(data_dir, directory)):
                if img[0] != '.':
                    orig = cv2.imread(os.path.join(data_dir, directory, img))
                    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(orig, 1.3, 5)
                    if len(faces) > 0:
                        (x, y, w, h) = faces[0]
                        max_area_face = faces[0]
                        for face in faces:
                            if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
                                max_area_face = face
                        face = max_area_face

                        # extract ROI of face
                        image = orig[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]

                        try:
                            # resize the image so that it can be passed to the neural network
                            image = cv2.resize(image, (48,48))
                            cv2.imwrite(os.path.join(face_dir, directory, img), image)
                        except Exception:
                            print("----->Problem during resize")
