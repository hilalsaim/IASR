"""Webcam utilities."""



import cv2

from facerecognition import gfxy




KEY_ESC = 27


def capture():
    vc = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    if vc.isOpened():
        _, frame = vc.read()
        return gfxy.Image(frame)


def display():
    vc = cv2.VideoCapture(0)
    key = 0
    success = True

    face_detector = gfxy.FaceDetector()

    while success and key != KEY_ESC:
        success, frame = vc.read()
        face_detector.show(gfxy.Image(frame), wait=False)
        key = cv2.waitKey(20)
