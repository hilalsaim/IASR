import cv2
import os

from src.test import predictFace
from src.test import detect_face


def openCamera():
    faces_dir = 'cropped_Faces/'
    videoCapture = cv2.VideoCapture(0)
    path_to_img = os.path.join(faces_dir, 's1', '1.pgm')

    img = detect_face(path_to_img)

    predictedName = predictFace(img)
    while True:
        # Capture frame-by-frame
        ret, frame = videoCapture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.3, 5)
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, predictedName, (x, y), cv2.FONT_HERSHEY_PLAIN, 2,
                    (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Face Recognation', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    videoCapture.release()
    cv2.destroyAllWindows()

openCamera()

