from os import listdir
from os.path import join, isfile

from PIL import Image
from numpy import array

TRAIN_PATH = 'BSR/BSDS500/data/images/train/'
TEST_PATH = 'BSR/BSDS500/data/images/test/'
GROUND_TRUTH_TEST_PATH = "BSR/BSDS500/data/groundTruth/test/"
TRUTH_SEG_COUNT = 4  # Number of different segmentations per image in GroundTruth


def read_image(path):
    img = Image.open(path)
    return array(img.getdata())


def read_ground_truth(folder_path):
    from scipy.io import loadmat
    import numpy
    # import pylab as plt

    extension = ".mat"

    file_names = [x for x in listdir(folder_path)]
    file_paths = [join(folder_path, x) for x in listdir(folder_path)]
    image_paths = [x for x in file_paths if isfile(x) and x.endswith(extension)]

    segmented_data = {}

    for image_name, image_path in zip(file_names, image_paths):
        mat = loadmat(image_path)
        im = mat["groundTruth"]
        images: numpy.ndarray = im

        table_key = image_name.replace(extension, "")
        segmented_data[table_key] = []

        for i in range(TRUTH_SEG_COUNT):
            # Couldn't make it pretty
            image = images[0][i][0][0][0]
            segmented_data[table_key].append(image)

            # Use for testing
            # plt.imshow(image)
            # plt.show()

    return segmented_data


if __name__ == "__main__":
    read_ground_truth(GROUND_TRUTH_TEST_PATH)
