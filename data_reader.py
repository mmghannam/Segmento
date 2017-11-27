from glob import glob
from PIL import Image
from numpy import array
from os import listdir
from os.path import join, isfile

TRAIN_PATH = 'BSR/BSDS500/data/images/train/'
TEST_PATH = 'BSR/BSDS500/data/images/test/'
GROUND_TRUTH_TEST_PATH = "BSR/BSDS500/data/groundTruth/test/"


def read_image(path):
    img = Image.open(path)
    return array(img.getdata())


def read_ground_truth(folder_path):
    from scipy.io import loadmat
    extension = ".mat"
    import numpy
    import matplotlib.pyplot as plt

    file_names = [x for x in listdir(folder_path)]
    file_paths = [join(folder_path, x) for x in listdir(folder_path)]
    image_paths = [x for x in file_paths if isfile(x) and x.endswith(extension)]

    segmented_data = {}

    for image_name, image_path in zip(file_names, image_paths):
        segmented_data[image_name] = []

        mat = loadmat(image_path)
        im = mat["groundTruth"]
        images: numpy.ndarray = im

        for i in range(4):
            image = images[0][i][0][0][0]
            segmented_data[image_name].append(image)

            # Use for testing
            # plt.imshow(image)
            # plt.show()

    return segmented_data


if __name__ == "__main__":
    read_ground_truth(GROUND_TRUTH_TEST_PATH)
