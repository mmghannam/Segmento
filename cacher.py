from data_reader import read_image
from kmeans import KMeans
from pickle import dump, load
from os.path import isfile


def cache_kmeans_results(files, target_folder, loc=False):
    for i,file in enumerate(files):
        print(i)
        image_data = read_image(file, with_location=loc)
        for k in range(3, 12, 2):
            cache_file_name = target_folder + file.split('/')[-1].replace('.jpg', '') + "-" + str(k)
            if not isfile(cache_file_name):
                with open(cache_file_name, "wb") as f:
                    cl = KMeans(image_data, k=k, tol=1)
                    result = cl.assign()
                    dump(result, f)


if __name__ == '__main__':
    from glob import glob

    files = glob('BSR/BSDS500/data/images/test/*.jpg')
    cache_kmeans_results(files, 'kmeans-loc-cache/', True)
