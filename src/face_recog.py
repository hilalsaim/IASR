import numpy as np
from src import img_read

image_width = 100
image_length = 100
total_pixels = image_width*image_length

images = 5
variants = 5
total_images = images*variants

face_vector = []

face_vector=img_read.X_train;
print(face_vector.shape)
avg_face_vector = face_vector.mean(axis=1)
avg_face_vector = avg_face_vector.reshape(face_vector.shape[0], 1)
normalized_face_vector = face_vector - avg_face_vector
covariance_matrix = np.cov(np.transpose(normalized_face_vector))

eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

k = 20
k_eigen_vectors = eigen_vectors[0:k, :]

eigen_faces = k_eigen_vectors.dot(np.transpose(normalized_face_vector))

weightst = np.transpose(normalized_face_vector).dot(np.transpose(eigen_faces))
print('Done with Train')


test_img=img_read.X_test
test_img = test_img.reshape(total_pixels, 1)
test_normalized_face_vector = img_read.X_test - avg_face_vector
test_weightt = np.transpose(test_normalized_face_vector).dot(np.transpose(eigen_faces))
print('Done with Test')
test_weight=test_weightt.transpose()
weights=weightst.transpose()
a=np.square(test_weight-weights)

index =  np.argmin(np.linalg.norm(a,axis=1 )) 
#b=np.sum(np.square(a))
#index = np.argmin(b)
#
if(index>=0 and index <5):
    print("S1")
if(index>=5 and index<10):
    print("S2")
if(index>=10 and index<15):
    print("S3")
if(index>=15 and index<20):
    print("S4")
if(index>=20 and index<25):
    print("S5")
if(index>=25):
    print('Unknown')
