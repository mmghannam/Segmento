from numpy import mean, array
from helpers import sum_square_error, euclidean_distance, manhattan_distance, resize_image
from data_reader import read_image, TRAIN_PATH, TEST_PATH
from sklearn.cluster import KMeans as K


class KMeans:
    @staticmethod
    def assign(data, k, tol, distance_func=manhattan_distance, max_iter=10):
        import random
        t = 0
        assignment = [0 for _ in range(len(data))]
        means = {}  # map between mean and cluster number
        last_error = 0
        for i in range(k):
            means[tuple(random.choice(data))] = i
        from time import time
        while True:
            start = time()
            t += 1
            # cluster assignment step
            startca = time()
            for i in range(len(data)):
                sample = data[i]
                closest_mean = min(means, key=lambda x: distance_func(array(x), sample))
                assignment[i] = means[closest_mean]
            print("Cluster Assignment took:", time() - startca, "seconds.")

            startcu = time()
            # centroid update step
            for last_mean in means.keys():
                class_instances = array([data[i] for i in range(len(data)) if assignment[i] == means[last_mean]])
                new_mean = class_instances.mean(axis=0)
                means[tuple(new_mean)] = means.pop(last_mean)
            print("Centroid Update took:", time() - startcu, "seconds.")

            starterr = time()
            new_error = sum_square_error(data, means, assignment)
            print("Error Calculation took:", time() - starterr, "seconds.")

            print("Iteration took:", time() - start, "seconds.")
            print('--------------------------------------------')
            if t > max_iter or abs(last_error - new_error) < tol:
                break
            last_error = new_error

        new_data = []
        means = list(means.keys())
        for i in assignment:
            new_data.append(tuple(int(x) for x in means[i]))

        return assignment, new_data


from PIL import Image
image_name = '55075.jpg'
img = Image.open(TRAIN_PATH + image_name)
img = resize_image(TRAIN_PATH + image_name, img.size[0] // 2, img.size[1] // 2)
print(img.size)
image_data = img.getdata()
new_image = Image.new(img.mode, img.size)
new_image_data = KMeans.assign(image_data, k=7, tol=1)[1]
new_image.putdata(new_image_data)
new_image.show()
