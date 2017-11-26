import numpy as np
import math
from kmeans import KMeans
from data_reader import TRAIN_PATH
from helpers import resize_image, euclidean_distance, show_image_from_data
from PIL import Image
from sklearn.preprocessing import normalize


class NCut:
    RBF_KERNEL = 0
    KNN = 1

    def __init__(self, data, n_clusters, gamma=0, n_neighbors=0, algorithm=RBF_KERNEL):
        self.data = data
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.n_clusters = n_clusters
        self.algorithm = algorithm
        self.sim_matrix = None
        self.n_cut_vector = None
        self.new_image_data = None

    def assign(self):
        if self.algorithm == NCut.RBF_KERNEL:
            self.sim_matrix = self.__make_rbf_sim_matrix(self.data, self.gamma)
        elif self.algorithm == NCut.KNN:
            self.sim_matrix = self.__make_knn_sim_matrix(self.data, self.n_neighbors)
        self.n_cut_vector = self.__make_normalized_cut(self.sim_matrix, self.n_clusters)
        self.new_image_data = self.__k_means_on_cut(self.n_cut_vector, self.n_clusters)
        return self.new_image_data

    @staticmethod
    def __make_rbf_sim_matrix(data, gamma):
        size = len(data)
        sim_matrix = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                sim_matrix[i, j] = math.exp(-gamma * (euclidean_distance(data[i], data[j]) ** 2))
        return sim_matrix

    @staticmethod
    def __make_knn_sim_matrix(data, n_neighbors):
        size = len(data)
        sim_matrix = np.zeros((size, size))

        # TODO(1): Generate KNN sim matrix

        return sim_matrix

    @staticmethod
    def __make_normalized_cut(sim_matrix, n_clusters):
        degree_matrix = np.zeros(sim_matrix.shape)
        for i in range(degree_matrix.shape[0]):
            degree_matrix[i, i] = sum(sim_matrix[i]) - 1
        eig_values, eig_vectors = np.linalg.eigh(1 - np.linalg.pinv(degree_matrix).dot(sim_matrix))
        idx = eig_values.argsort()
        eig_vectors = eig_vectors[:, idx]
        return normalize(eig_vectors[:, :n_clusters], axis=1, norm='l1')

    @staticmethod
    def __k_means_on_cut(eig_vectors, clusters):
        final_clustering = KMeans(eig_vectors, k=clusters, tol=1)
        return final_clustering.assign()[1]


if __name__ == "__main__":
    image_name = '55075.jpg'
    path = TRAIN_PATH + image_name
    img = Image.open(path)
    img = resize_image(path, img.size[0] // 10, img.size[1] // 10)
    image_data = img.getdata()
    clustering = NCut(image_data, 3, gamma=10, algorithm=NCut.RBF_KERNEL)
    new_image_data = clustering.assign()
    show_image_from_data(img.mode, img.size, new_image_data)
