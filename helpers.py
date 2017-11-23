from numpy.linalg import norm


def euclidean_distance(a, b):
    return norm(a - b)


def manhattan_distance(a, b):
    return sum(abs(xn - yn) for xn, yn in zip(a, b))


def sum_square_error(assignment):
    error = 0
    for mean in assignment.keys():
        for element in assignment[mean]:
            error += norm(element - mean) ** 2
    return error
