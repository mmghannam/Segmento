from collections import Counter
import numpy as np


def conditional_entropy(result, truth):
    entropy = 0

    for cluster in np.unique(result):
        cluster_elements = truth[np.where(result == cluster)]
        elements_counter = Counter(cluster_elements)

        cluster_entropy = 0
        for _count in elements_counter.keys():
            coeff = elements_counter[_count] / np.size(result)
            probability = elements_counter[_count] / np.size(cluster_elements)

            cluster_entropy -= coeff * np.log2(probability)

        entropy += cluster_entropy

    return entropy


def f_measure(result, truth):
    truth_counter = Counter(truth)
    f_measure = list()
    for cluster in np.unique(result):
        cluster_elements = truth[np.where(result == cluster)]
        cluster_count = Counter(cluster_elements)
        max_partition = cluster_count.most_common(1)[0]
        purity = max_partition[1] / cluster_elements.size
        recall = max_partition[1] / truth_counter[max_partition[0]]
        f_measure.append(2 * purity * recall / (purity + recall))
    return sum(f_measure) / len(f_measure)


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


def class_equality(prediction, truth):
    """
    returns the counts of each pair of assignment
    """
    result = {}
    for p, t in zip(prediction, truth):
        if p not in result:
            result[p] = {t: 1}
        elif t not in result[p]:
            result[p][t] = 1
        else:
            result[p][t] += 1
    return result


def best_equal_clusters(prediction, truth):
    """
    returns a dict of what assignment classes are equal to each other
    """
    cl_eq = class_equality(prediction, truth)
    result = {}
    for cl in cl_eq.keys():
        cl_counts = cl_eq[cl]
        try:
            result[cl] = max([cl for cl in cl_eq[cl].keys() if cl not in result], key=lambda x: cl_counts[x])
        except:
            # happens if no max to be found (empty list)
            pass
    return result


def both_assignment_the_same(prediction, truth):
    bec = best_equal_clusters(prediction, truth)
    print(bec)
    for i, element in enumerate(prediction):
        try:
            prediction[i] = bec[element]
        except KeyError:
            continue
    return prediction, truth


def evaluate_segmentation_from_cache(segmentation_technique, eval_func):
    from glob import glob
    from pickle import load, dump
    from data_reader import read_ground_truth, GROUND_TRUTH_TEST_PATH
    from os.path import isfile

    ground_truth = read_ground_truth(GROUND_TRUTH_TEST_PATH)
    CACHED_RESULTS_PATH = segmentation_technique + '-cache/'
    EVALUATION_RESULTS_PATH = segmentation_technique + '-evaluations-' + eval_func.__name__ + '/'
    for k in range(3, 12, 2):
        for result_file_name in glob(CACHED_RESULTS_PATH + '*-' + str(k)):
            print(result_file_name, k)
            img_name = result_file_name.replace(CACHED_RESULTS_PATH, '').split('-')[0]
            with open(result_file_name, 'rb') as f:
                result_assignment = load(f)[0]

            cached_file_name = EVALUATION_RESULTS_PATH + img_name + '-' + str(k)
            if not isfile(cached_file_name):
                evaluations = []
                for truth in ground_truth[img_name]:
                    evaluation = eval_func(result_assignment, truth.flatten())
                    evaluations.append(evaluation)
                with open(cached_file_name, 'wb') as f:
                    dump(evaluations, f)


def print_evaluations(cached_files):
    from pickle import load
    for file in sorted(cached_files):
        print('file name:', file.split('/')[-1])
        with open(file, 'rb') as f:
            evaluations = load(f)
            average = sum(evaluations) / len(evaluations)
            print('average evaluation:', average)
            print('----------------------------')


if __name__ == '__main__':
    from glob import glob

    # evaluate_segmentation_from_cache('kmeans', conditional_entropy)
    # print(len(glob('kmeans-evaluations-conditional_entropy/*')))
    # print_evaluations(glob('kmeans-evaluations-conditional_entropy/*'))


    evaluate_segmentation_from_cache('kmeans', f_measure)
    print(len(glob('kmeans-evaluations-f_measure/*')))
