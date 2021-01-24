from PIL import Image
import os
import os.path
import numpy as np

class EigenFaces(object):
    def train(self):
        self.projected_classes = []

        self.list_of_arrays_of_images, self.labels_list, \
            list_of_matrices_of_flattened_class_samples = \
                read_images()

         # create matrix to store all flattened images
        images_matrix = np.array([np.array(Image.fromarray(img)).flatten()
              for img in self.list_of_arrays_of_images],'f')

        # perform PCA
        self.eigenfaces_matrix, variance, self.mean_image = PCA(images_matrix)

        # Projecting each class sample (as class matrix) and then using the class average as the class weights for comparison with the Target image
        for class_sample in list_of_matrices_of_flattened_class_samples:
            class_weights_vertex = self.project_image(class_sample)
            self.projected_classes.append(class_weights_vertex.mean(0))

    def project_image(self, X):
        X = X - self.mean_image
        return np.dot(X, self.eigenfaces_matrix.T)

    def predict_face(self, X):
        min_class = -1
        min_distance = np.finfo('float').max
        projected_target = self.project_image(X)
        # delete last array item, it's nan
        projected_target = np.delete(projected_target, -1)
        for i in range(len(self.projected_classes)):
            distance = np.linalg.norm(projected_target - np.delete(self.projected_classes[i], -1))
            if distance < min_distance:
                min_distance = distance
                min_class = self.labels_list[i]
        # print(min_class, min_distance)
        return min_class

    def __repr__(self):
        return "PCA (num_components=%d)" % (self._num_components)


def read_images():
    class_matrices_list = []
    images, image_labels = [], []
    for dirname, dirnames, filenames in os.walk('cropped_Faces'):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            class_samples_list = []
            for filename in os.listdir(subject_path):
                    im = Image.open(os.path.join(subject_path, filename))
                    images.append(np.asarray(im, dtype = np.uint8))

                    # adds each sample within a class to this List
                    class_samples_list.append(np.asarray(im, dtype = np.uint8))

            # flattens each sample within a class and adds the array/vector to a class matrix
            class_samples_matrix = np.array([img.flatten()
                for img in class_samples_list],'f')

             # adds each class matrix to this MASTER List
            class_matrices_list.append(class_samples_matrix)

            image_labels.append(subdirname)

    return images, image_labels, class_matrices_list


def PCA(X):
    num_data, dimension = X.shape
    mean_X = X.mean(axis=0)
    X = X - mean_X

    if dimension > num_data:
        covariancematrix = np.dot(X,X.T)
        eigenvalues,eigenvectors = np.linalg.eigh(covariancematrix)
        tmp = np.dot(X.T,eigenvectors).T # this is the compact trick
        projectionmatrix = tmp[::-1] # reverse since last eigenvectors are the ones we want
        variance = np.sqrt(eigenvalues[::-1]) # reverse since eigenvalues are in increasing order

        for i in range(projectionmatrix.shape[1]):
            projectionmatrix[:,i] /= variance
    else:
        U, variance, projectionmatrix = np.linalg.svd(X)
        projectionmatrix = projectionmatrix[:num_data] # only makes sense to return the first num_data

    return projectionmatrix, variance, mean_X