from numpy import mean, array
from helpers import sum_square_error, euclidean_distance
from data_reader import read_image, TRAIN_PATH, TEST_PATH
from sklearn.cluster import KMeans as K


class KMeans:
    @staticmethod
    def assign(data, k, tol, distance_func=euclidean_distance, max_iter=100):
        import random
        t = 0
        assignment = [0 for _ in range(len(data))]
        means = {}  # map between mean and cluster number
        last_error = 0
        for i in range(k):
            means[tuple(random.choice(data))] = i

        while True:
            t += 1
            # cluster assignment step
            for i in range(len(data)):
                sample = data[i]
                closest_mean = min(means, key=lambda x: distance_func(array(x), sample))
                assignment[i] = means[closest_mean]

            # centroid update step
            for last_mean in means.keys():
                class_instances = array([data[i] for i in range(len(data)) if assignment[i] == means[last_mean]])
                new_mean = class_instances.mean(axis=0)
                means[tuple(new_mean)] = means.pop(last_mean)

            new_error = sum_square_error(data, means, assignment)
            if t > max_iter or abs(last_error - new_error) < tol:
                break
            last_error = new_error

        new_data = []
        means = list(means.keys())
        for i in assignment:
            new_data.append(tuple(int(x) for x in means[i]))
        return new_data


from PIL import Image

image_data = read_image(TRAIN_PATH + '16052.jpg')
img = Image.open(TRAIN_PATH + '16052.jpg')
new_image = Image.new(img.mode, img.size)
new_image_data = KMeans.assign(image_data, k=3, tol=1)
new_image.putdata(new_image_data)
new_image.show()
