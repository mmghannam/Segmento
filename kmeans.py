from numpy import array, allclose, amin
from copy import copy
from helpers import *
from data_reader import *
from time import time
import random
import numpy as np


class KMeans:
    def __init__(self, data, k, tol, max_iter=100):
        self.data = data
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def assign(self):
        t = 0
        means = [tuple(x) for x in random.sample(list(self.data), self.k)]
        last_means = [[0] * self.data.shape[1] for _ in range(self.k)]

        while True:
            start = time()
            t += 1
            # cluster assignment step
            assignment = self.cluster_assignment(means)
            # centroid update step
            self.centroid_update(means, assignment)

            if t > self.max_iter or allclose(means, last_means, atol=self.tol):
                break

            last_means = copy(means)

            print("Iteration", t, "took:", time() - start, "seconds.")
            print('--------------------------------------------')

        new_data = []
        means = list(means)
        for i in assignment:
            new_data.append(tuple(int(x) for x in means[i]))

        return assignment, new_data

    @timed
    def cluster_assignment(self, means):
        means = array(means)
        distances = np.abs(self.data - means[:, np.newaxis]).sum(axis=2)
        return np.argmin(distances, axis=0)

    @timed
    def centroid_update(self, means, assignment):
        for ki in range(self.k):
            class_instances = self.data[array(assignment) == ki]
            if len(class_instances) == 0:
                continue
            new_mean = class_instances.mean(axis=0)
            means[ki] = tuple(new_mean)

    @timed
    def error_calculation(self, means, assignment):
        return sum_square_error(self.data, means, assignment)


if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt

    images = ['81095.jpg', '2018.jpg', '3063.jpg', '5096.jpg', '6046.jpg']

    ground_truths = read_ground_truth(GROUND_TRUTH_TEST_PATH)
    for image_name in images:
        plt.imshow(ground_truths[image_name.replace('.jpg', '')][0])

    plt.show()

    for image_name in images:
        path = TEST_PATH + image_name
        img = Image.open(path)
        # img = resize_image(path, img.size[0] // 2, img.size[1] // 2)
        print(img.mode, img.size)
        image_data = array(img.getdata())
        new_image = Image.new(img.mode, img.size)
        clusterer = KMeans(image_data, k=5, tol=1)
        new_image_data = clusterer.assign()[1]
        show_image_from_data(new_image_data, img.mode, img.size )


