from glob import glob
from PIL import Image
from numpy import array

TRAIN_PATH = 'BSR/BSDS500/data/images/train/'
TEST_PATH = 'BSR/BSDS500/data/images/test/'


def read_image(path):
    img = Image.open(path)
    return array(img.getdata())
