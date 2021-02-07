import cv2
import os.path
import numpy as np
from sklearn.model_selection import train_test_split

faces_dir = 'dB/'

image_faces_count = 5  # number of faces used for training
faces_count = 5  # number of faces used for testing

l = image_faces_count * faces_count  # training images count
m = 64  # number of columns of the image
n = 64  # number of rows of the image
mn = m * n  # length of the column vector

print ('> Initializing started')
training_ids = []  # train image id's for every at&t face
L = np.empty(shape=(mn, l), dtype='float64')  # each row of L represents one train image
cur_img = 0

# check if the file exists, otherwise create it
def file_is_exists(path):
    if not os.path.isdir(path):
        os.mkdir(path)


# cropping face from image
def detect_face(img_path):
    img = cv2.imread(img_path)

    detected_faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') .detectMultiScale(img, 1.3, 5)
    x, y, w, h = detected_faces[0]  # focus on the 1st face in the image

    img = img[y:y + h, x:x + w]  # focus on the detected area
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


for face_id in range(1, faces_count + 1):

    cou1nt1 = 0
    training_ids = range(1, 6)

    for training_id in training_ids:
        path_to_img = os.path.join(faces_dir, 's' + str(face_id), str(training_id) + '.jpg')
        print('> reading file: ' + path_to_img)

        img2=detect_face(path_to_img)

        img_col = np.array(img2, dtype='float64').flatten()
        L[:, cur_img] = img_col[:]  # set the cur_img-th column to the current training image
        cur_img += 1
        print(cur_img)

        face_path = 'cropped_Faces/{}'.format("s"+"%d"%face_id)
        file_is_exists(face_path) # check if the file exists, otherwise create it

        cv2.imwrite("cropped_Faces/"+"s"+"%d"%face_id+"/%d.pgm" % training_id, img2) # save cropping face

        print ('> Initializing ended')

X_train, X_test = train_test_split(np.transpose(L), test_size=0.03)
