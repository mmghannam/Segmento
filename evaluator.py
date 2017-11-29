import math


def conditional_entropy(result, truth):
    # Input is a list, index:pixel , value: cluster
    entropy = 0
    result_counts = count_clusters(result)
    truth_counts = count_clusters(truth)

    for idx in truth_counts.keys():
        nij = result_counts[idx]  # Correct elements
        ni = truth_counts[idx]  # Deduced elements

        percentage = nij / ni
        entropy += percentage * math.log(percentage, 2)

    return -1 * entropy


def count_clusters(items):
    counts = {}

    for (i, cluster) in enumerate(items):
        if cluster not in counts:
            counts[cluster] = 1
        else:
            counts[cluster] += 1

    return counts
