import numpy as np
from numpy import *
import cv2
import os

from src import img_read

imageLabels=[]
faces_dir = 'cropped_Faces/'
face_vector = []
face_vector=img_read.X_train;

#File names list
for dirname, dirnames, filenames in os.walk('cropped_Faces'):
    for subdirname in dirnames:
        path = os.path.join(dirname, subdirname)
        imageLabels.append(subdirname)

#Principal Component Analysis, or PCA, is a dimensionality-reduction method
def pca(face_vector):
    avg_face_vector = face_vector.mean(axis=0)
    face_vector = face_vector - avg_face_vector

    num_observations, num_dimensions = face_vector.shape

    if num_dimensions > 100:
        eigenvalues, eigenvectors = linalg.eigh(dot(face_vector, face_vector.T))
        v = (dot(face_vector.T, eigenvectors).T)[::-1]
        s = sqrt(eigenvalues)[::-1]
    else:
        u, s, v = linalg.svd(face_vector, full_matrices=False)

    return v,avg_face_vector


eigenfacesMatrix, meanImage = pca(face_vector)



#predicting that face belongs to whom
def predictFace(X):
    min = -1
    mDistance = np.finfo('float').max
    X = X - meanImage

    for i in range(len(X)):
        distance = 0
        if distance < mDistance:
            mDistance = distance
            min= imageLabels[i]
    return min


#crop faces area
def detect_face(img_path):
    img = cv2.imread(img_path)

    detected_faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') .detectMultiScale(img, 1.3, 5)
    x, y, w, h = detected_faces[0]  # focus on the 1st face in the image

    img = img[y:y + h, x:x + w]  # focus on the detected area
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return np.array(img, dtype=np.uint8).flatten()



path_to_img = os.path.join(faces_dir, 's1','1.pgm')
img=detect_face(path_to_img)
predictedName = predictFace(img)


print(predictedName)
print(path_to_img)
print(imageLabels)
