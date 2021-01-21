# -*- coding: utf-8 -*-

import cv2

import os.path

import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image, ImageOps

from src.face_cropper import FaceCropper

faces_dir = 'dB/'

new_data_dir = 'dB/'

image_faces_count = 5  # number of faces used for training
faces_count = 5  # number of faces used for testing

l = image_faces_count * faces_count  # training images count
m = 32  # number of columns of the image
n = 32  # number of rows of the image
mn = m * n  # length of the column vector

print ('> Initializing started')
training_ids = []  # train image id's for every at&t face
L = np.empty(shape=(mn, l), dtype='float64')  # each row of L represents one train image
cur_img = 0

for face_id in range(1, faces_count + 1):

    cou1nt1 = 0
    training_ids = range(1, 6)

    for training_id in training_ids:
        path_to_img = os.path.join(faces_dir, 's' + str(face_id), str(training_id) + '.jpg')
        print('> reading file: ' + path_to_img)

        detecter = FaceCropper()
        img2 = detecter.generate(path_to_img, True)
        cv2.imwrite("FaceDataNew/"+"s"+"%d"%face_id+"/%d.jpg" % training_id, img2)#creating new image




        img1 = Image.open(path_to_img)

        img = ImageOps.grayscale(img1)
        resiz = (32, 32)
        img2 = img2.resize(resiz)
        img_col = np.array(img2, dtype='float64').flatten()
        L[:, cur_img] = img_col[:]  # set the cur_img-th column to the current training image
        cur_img += 1
        print(cur_img)
        print ('> Initializing ended')

X_train, X_test = train_test_split(np.transpose(L), test_size=0.03)


