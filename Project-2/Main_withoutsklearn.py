import numpy as np

from timeit import default_timer
import sys


def readVectorsSeq(filename):
    with open(filename) as f:
        result = [tuple(map(float, i.split(','))) for i in f]
    return result


def euclidean(point1, point2):
    """ Calculate the euclidean norm distance between two points """
    return np.linalg.norm(point1 - point2)


def get_ball_indices(idxs, distance_matrix, x_idx, radius):
    """ Write the docstring """
    return [i for i in idxs if distance_matrix[x_idx][i] <= radius and x_idx != i]


def calc_ball_weight(idx, z_idxs, weights, distance_matrix, radius):
    """ Write the docstring """
    return np.sum([weights[z_index] for z_index in z_idxs if distance_matrix[z_index][idx] <= radius])


def find_new_center_idx(input_size, Z_idxs, weights, distance_matrix, radius):
    """ Write the docstring """
    ball_weights = [
        calc_ball_weight(idx=i, z_idxs=Z_idxs, weights=weights, distance_matrix=distance_matrix, radius=radius) for i in
        range(input_size)]
    # return index of maximum value
    return ball_weights.index(np.max(ball_weights))


def find_ball_indices(idx, idxs, distance_matrix, radius):
    """ Write the docstring """
    return [z_idx for z_idx in idxs if distance_matrix[idx][z_idx] <= radius]


def pairwise_distances(P: np.ndarray) -> np.ndarray:
    """ Return the matrix with the pairwise distances of hte points in the initial array"""
    points_nr = len(P)
    # Initialize distance matrix
    distance_matrix = np.zeros((points_nr, points_nr), dtype=float)
    # Loop to calculate the tri-diagonal upper matrix
    for i in range(points_nr):
        for j in range(i + 1, points_nr):
            distance_matrix[i, j] = euclidean(P[i], P[j])
    # Fill the tri-diagonal lower matrix
    distance_matrix += distance_matrix.T
    return distance_matrix


def SeqWeightedOutliers(P, W, k, z, alpha):
    """ Write the docstring """
    start = default_timer()

    # calc r_initial
    subset = P[: k + z + 1]
    distance_matrix = pairwise_distances(subset)
    r = r_initial = np.min(distance_matrix[distance_matrix != 0]) / 2

    # Calculate the matrix distance between all the points
    distances = pairwise_distances(P)

    n_guesses = 1
    input_size = len(P)
    while True:
        Z_idxs = [i for i in range(len(P))]
        S_idxs = []
        Wz = np.sum(W)
        while (len(S_idxs) < k) and (Wz > 0):
            new_center_idx = find_new_center_idx(input_size, Z_idxs, W, distances, radius=(1 + 2 * alpha) * r)

            S_idxs.append(new_center_idx)

            Bz_indices = find_ball_indices(idx=new_center_idx,
                                           idxs=Z_idxs,
                                           distance_matrix=distances,
                                           radius=(3 + 4 * alpha) * r)

            for Bz_index in Bz_indices:
                Z_idxs.remove(Bz_index)
                Wz -= W[Bz_index]

        if Wz <= z:
            end = default_timer()

            print(f'Input size n = {input_size}')
            print(f'Number of centers k = {k}')
            print(f'Number of outliers z = {z}')
            print(f'Initial guess = {r_initial}')
            print(f'Final guess = {r}')
            print(f'Number of guesses = {n_guesses}')
            print(f'Objective function = {ComputeObjective(P, P[S_idxs], len(Z_idxs))}')
            print(f'Time of SeqWeightedOutliers = {(end - start) * 1000}')

            return P[S_idxs]
        else:
            r *= 2
            n_guesses += 1


def ComputeObjective(P, S, z):
    dists = [np.min([euclidean(x, center) for center in S]) for x in P]
    dists.sort(reverse=True)
    return np.max(dists[z:])


def main(argv):
    file_path = argv[1]
    k = int(argv[2])
    z = int(argv[3])

    # Read data
    inputPoints = np.array(readVectorsSeq(file_path), dtype=float)

    # Initialize the weights
    weights = np.ones((len(inputPoints),), dtype=float)
    # Calculate the solution
    solution = SeqWeightedOutliers(inputPoints, weights, k, z, alpha=0)


if __name__ == '__main__':
    main(sys.argv)
