from numpy import mean, array
from helpers import sum_square_error, euclidean_distance, manhattan_distance, resize_image, show_image_from_data, timed
from data_reader import read_image, TRAIN_PATH, TEST_PATH
from sklearn.cluster import KMeans as K
from time import time


class KMeans:
    def __init__(self, data, k, tol, distance_func=manhattan_distance, max_iter=10):
        self.data = data
        self.k = k
        self.tol = tol
        self.distance_func = distance_func
        self.max_iter = max_iter

    def assign(self):
        import random
        t = 0
        assignment = [0 for _ in range(len(self.data))]
        means = {}  # map between mean and cluster number
        last_error = 0

        for i in range(self.k):
            means[tuple(random.choice(self.data))] = i

        while True:
            start = time()
            t += 1
            # cluster assignment step
            self.cluster_assignment(means, assignment)

            # centroid update step
            self.centroid_update(means, assignment)

            new_error = self.error_calculation(means, assignment)

            print("Iteration", t, "took:", time() - start, "seconds.")
            print('--------------------------------------------')

            if t > self.max_iter or abs(last_error - new_error) < self.tol:
                break

            last_error = new_error

        new_data = []
        means = list(means.keys())
        for i in assignment:
            new_data.append(tuple(int(x) for x in means[i]))

        return assignment, new_data

    @timed
    def cluster_assignment(self, means, assignment):
        for i in range(len(self.data)):
            sample = self.data[i]
            closest_mean = min(means, key=lambda x: self.distance_func(array(x), sample))
            assignment[i] = means[closest_mean]

    @timed
    def centroid_update(self, means, assignment):
        for last_mean in means.keys():
            class_instances = array([self.data[i] for i in range(len(self.data)) if assignment[i] == means[last_mean]])
            new_mean = class_instances.mean(axis=0)
            means[tuple(new_mean)] = means.pop(last_mean)

    @timed
    def error_calculation(self, means, assignment):
        return sum_square_error(self.data, means, assignment)


if __name__ == "__main__":
    from PIL import Image
    image_name = '55075.jpg'
    path = TRAIN_PATH + image_name
    img = Image.open(path)
    # img = resize_image(path, img.size[0] // 2, img.size[1] // 2)
    image_data = img.getdata()
    new_image = Image.new(img.mode, img.size)
    clusterer = KMeans(image_data, k=11, tol=1)
    new_image_data = clusterer.assign()[1]
    show_image_from_data(img.mode, img.size, new_image_data)
