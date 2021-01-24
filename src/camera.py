import cv2
from src import imageFunctions

def cameraCapture():
    vc = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    if vc.isOpened():
        _, frame = vc.read()
        return imageFunctions.Image(frame)

def openCamera():
    webcamera = cv2.VideoCapture(0)
    success = True

    face_detector = imageFunctions.FaceDetector()

    while success:
        success, frame = webcamera.read()
        face_detector.show(imageFunctions.Image(frame), wait=False)
        cv2.waitKey(20)