import cv2
from src import imageFunctions
import os

def dir_exists(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def training():
    name = input('Enter your name: ')
    take_training_photos(name, 10)

def cameraCapture():
    vc = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    if vc.isOpened():
        _, frame = vc.read()
        return imageFunctions.Image(frame)

def take_training_photos(name, n):
    for i in range(n):
        for face in cameraCapture().faces():
            normalized = face.gray().scale(100, 100)

            face_path = 'cropped_Faces/{}'.format(name)
            dir_exists(face_path)
            normalized.save_to('{}/{}.pgm'.format(face_path, i + 1))

            normalized.show()

training()