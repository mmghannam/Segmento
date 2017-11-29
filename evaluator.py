import math


def conditional_entropy(result, truth):
    # Input is a list, index:pixel , value: cluster
    entropy = 0
    result_counts = count_correctly_clustered(result, truth)
    truth_counts = count_clusters(truth)

    for idx in truth_counts.keys():
        nij = result_counts[idx]  # Correct elements
        ni = truth_counts[idx]  # Deduced elements

        percentage = nij / ni
        entropy += percentage * math.log(percentage, 2)

    return -1 * entropy


def f_measure(result, truth):
    # Ni is the number of points in a cluster
    result_cluster_stats = count_correctly_clustered(result, truth)
    truth_cluster_stats = count_clusters(truth)

    purity = calculate_purity(result_cluster_stats, truth_cluster_stats)
    raise NotImplementedError()


def calculate_purity(result_cluster_stats, truth_cluster_stats):
    purity = 0
    # Sum of values in the truth is the total number of pixels
    total_pixel_count = sum(truth_cluster_stats.values())

    for value in result_cluster_stats.values():
        purity += value / total_pixel_count

    return purity


def count_correctly_clustered(result, truth):
    """
    Computes the value n_ij for a given clustering and its ground truth
    the n_ij value is the number of the truly identified pixels
    :param result: Calculated clustering
    :param truth: Ground truth clustering
    :return: Nij for all clusters in a hashtable
    """
    # We shouldn't check for programmers mistakes, but meh,
    # If this throws then the images might be differnet
    assert len(result) == len(truth)

    counts = {}

    for x, y in zip(result, truth):
        if x != y:
            continue

        # Create an entry in the table
        if x not in counts:
            counts[x] = 0

        counts[x] += 1

    return counts


def count_clusters(items):
    """
    Counts the elements in all clusters in a given clustering
    :param items: Result of the clustering process
    """
    counts = {}

    for (i, cluster) in enumerate(items):
        if cluster not in counts:
            counts[cluster] = 1
        else:
            counts[cluster] += 1

    return counts
