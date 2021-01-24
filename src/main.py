import sys
import os
import cv2
from src import camera


def ensure_dir_exists(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def take_training_photos(name, n):
    for i in range(n):
        for face in camera.cameraCapture().faces():
            normalized = face.gray().scale(100, 100)

            face_path = 'cropped_Faces/{}'.format(name)
            ensure_dir_exists(face_path)
            normalized.save_to('{}/{}.pgm'.format(face_path, i + 1))

            normalized.show()


def parse_command():
    args = sys.argv[1:]
    return args[0] if args else None

def training():
    name = input('Enter your name: ')
    take_training_photos(name, 10)

def main():
    cmd = parse_command()
    if cmd == 'train':
        training()
    elif cmd == 'demo':
        camera.openCamera()

if __name__ == "__main__":
    main()