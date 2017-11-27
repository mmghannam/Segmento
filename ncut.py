import numpy as np
import math
from random import randint
from sklearn.cluster import KMeans
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
        self.new_image_data = self.__k_means_on_cut(self.data, self.n_cut_vector, self.n_clusters)
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
        eig_vectors = np.linalg.eigh(degree_matrix - sim_matrix)[1]
        return normalize(eig_vectors[:, :n_clusters], axis=1, norm='l1')

    @staticmethod
    def __k_means_on_cut(data, eig_vectors, clusters):
        new_data = []
        cluster_colors = []
        final_clustering = KMeans(clusters, n_jobs=-1)
        assignments = final_clustering.fit_predict(eig_vectors)
        for _ in range(clusters):
            cluster_colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
        for i in range(len(data)):
            new_data.append(cluster_colors[assignments[i]])
        return new_data


if __name__ == "__main__":
    image_name = '135069.jpg'
    path = TRAIN_PATH + image_name
    img = Image.open(path)
    img = resize_image(path, img.size[0] // 5, img.size[1] // 5)
    image_data = img.getdata()
    show_image_from_data(img.mode, img.size, image_data)
    clustering = NCut(image_data, 2, gamma=1, algorithm=NCut.RBF_KERNEL)
    new_image_data = clustering.assign()
    show_image_from_data(img.mode, img.size, new_image_data)
