from helpers import sum_square_error, euclidean_distance


class KMeans:
    def assign(data, k, error, distance_func=euclidean_distance, max_iter=100):
        import random
        t = 0
        assignment = {}
        for _ in range(k):
            assignment[tuple(random.choice(data))] = []

        while True:
            t += 1
            # cluster assignment step
            for sample in data:
                closest_mean = min(assignment.keys(), key=lambda x: distance_func(array(x), sample))
                assignment[closest_mean].append(sample)
            # centroid update step
            for last_mean in [key for key in assignment.keys()]:
                new_mean = mean(assignment[last_mean], axis=0)
                assignment[tuple(new_mean)] = assignment[last_mean]
                del assignment[last_mean]

            if sum_square_error(assignment) < error or t > max_iter:
                break
        return assignment
