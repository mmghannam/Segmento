import math


def conditional_entropy(k, ni, nij):
    entropy = 0

    for idx in range(k):
        percentage = nij[idx] / ni[idx]
        entropy += percentage * math.log(percentage, 2)

    return -1 * entropy


def f_measure(, k, ni, nij):
    pass
