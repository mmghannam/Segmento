import numpy as np
import math
from data_reader import TRAIN_PATH
from helpers import resize_image, euclidean_distance
from PIL import Image
import matplotlib.pyplot as plt


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

    def assign(self):
        if self.algorithm == NCut.RBF_KERNEL:
            self.sim_matrix = self.__make_rbf_sim_matrix(self.data, self.gamma)
        elif self.algorithm == NCut.KNN:
            self.sim_matrix = self.__make_knn_sim_matrix(self.data, self.n_neighbors)
        plt.imshow(self.sim_matrix)
        plt.title('Similarity Matrix')
        plt.show()
        self.__make_normalized_cut(self.data, self.sim_matrix, self.n_clusters)

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
    def __make_normalized_cut(data, sim_matrix, clusters):
        pass
        # TODO(2): Implement NCut


image_name = '55075.jpg'
path = TRAIN_PATH + image_name
img = Image.open(path)
img = resize_image(path, img.size[0] // 10, img.size[1] // 10)
image_data = img.getdata()
clustering = NCut(image_data, 5, gamma=1, algorithm=NCut.RBF_KERNEL)
clustering.assign()
