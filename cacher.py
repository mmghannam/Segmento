from data_reader import read_image
from kmeans import KMeans
from pickle import dump, load
from os.path import isfile


def cache_kmeans_results(files, target_folder):
    for file in files:
        image_data = read_image(file)
        for k in range(3, 11, 2):
            cache_file_name = target_folder + file.split('/')[-1].replace('.jpg', '') + "-" + str(k)
            if not isfile(cache_file_name):
                with open(cache_file_name, "wb") as f:
                    cl = KMeans(image_data, k=k, tol=1)
                    result = cl.assign()
                    dump(result, f)


if __name__ == '__main__':
    from glob import glob

    files = glob('BSR/BSDS500/data/images/test/*')
    cache_kmeans_results(files, 'kmeans-cache/')
