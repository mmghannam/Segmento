from numpy.linalg import norm


def euclidean_distance(a, b):
    return norm(a - b)


def manhattan_distance(a, b):
    return sum(abs(xn - yn) for xn, yn in zip(a, b))


def sum_square_error(data, means, assignment):
    error = 0
    for mean in means.keys():
        for element in (data[i] for i in range(len(data)) if assignment[i] == means[mean]):
            error += norm(element[1] - mean[1]) ** 2
    return error


def resize_image(path, x, y):
    from PIL import Image

    with open(path, 'r+b') as f:
        with Image.open(f) as image:
            image = image.resize((x, y), Image.NEAREST)
            return image
