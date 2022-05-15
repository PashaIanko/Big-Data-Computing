from sklearn.metrics import pairwise_distances
from os.path import isfile

from numpy import min
import numpy as np
from numpy import sum
from numpy import array
from timeit import default_timer
import sys
from math import sqrt


def readVectorsSeq(filename):
    with open(filename) as f:
        result = [tuple(map(float, i.split(','))) for i in f]
    return result


def euclidean(point1, point2):
    return np.linalg.norm(point1 - point2)
    # res = 0
    # for i in range(len(point1)):
    #     diff = (point1[i] - point2[i])
    #     res += diff * diff
    # return sqrt(res)


def get_ball_indices(idxs, distance_matrix, x_idx, radius):
    return [i for i in idxs if distance_matrix[x_idx][i] <= radius and x_idx != i]
    # res = []
    # for i in idxs:
    #     # print(f'Dist: {distance_matrix[x_idx][i]}, radius: {radius}')
    #     if distance_matrix[x_idx][i] <= radius and x_idx != i:
    #         res.append(i)
    # return res

def calc_ball_weight(idx, z_idxs, weights, distance_matrix, radius):
    return sum([weights[z_index] for z_index in z_idxs if distance_matrix[z_index][idx] <= radius])

    # weight = 0
    # for z_idx in z_idxs:
    #     if distance_matrix[z_idx][idx] <= radius:
    #         weight += weights[z_idx]
    # return weight


def find_new_center_idx(input_size, Z_idxs, weights, distance_matrix, radius):
    ball_weights = [calc_ball_weight(idx=i, z_idxs=Z_idxs, weights=weights, distance_matrix=distance_matrix, radius=radius) for i in range(input_size)]
    # return index of maximum value
    return ball_weights.index(max(ball_weights))

    # max_weight = 0
    # new_center_idx = None
    # for i in P_idxs:
    #     ball_weight = calc_ball_weight(
    #         idx=i,
    #         z_idxs=Z_idxs,
    #         weights=weights,
    #         distance_matrix=distance_matrix,
    #         radius=radius
    #     )
    #     if ball_weight > max_weight:
    #         max_weight = ball_weight
    #         new_center_idx = i
    # return new_center_idx


def find_ball_indices(idx, idxs, distance_matrix, radius):
    return [z_idx for z_idx in idxs if distance_matrix[idx][z_idx] <= radius]
    # res = []
    # for z_idx in idxs:
    #     if distance_matrix[idx][z_idx] <= radius:
    #         res.append(z_idx)
    # return res


def SeqWeightedOutliers(P, W, k, z, alpha):
    start = default_timer()

    # calc r_initial
    subset = P[: k + z + 1]
    r_dist = pairwise_distances(subset)
    r = np.min(r_dist[r_dist != 0]) / 2
    r_initial = r

    distances = pairwise_distances(P)
    n_guesses = 1
    input_size = len(P)
    while(True):
        # P_idxs = [i for i in range(len(P))]

        Z_idxs = [i for i in range(len(P))]
        S_idxs = []
        Wz = sum(W)
        while (len(S_idxs) < k) and (Wz > 0):
            new_center_idx = find_new_center_idx(input_size, Z_idxs, W, distances, radius=(1 + 2 * alpha) * r)
            # assert(not (new_center_idx is None))
            # assert(not (new_center_idx in S_idxs))

            S_idxs.append(new_center_idx)

            Bz_indices = find_ball_indices(idx=new_center_idx, idxs=Z_idxs, distance_matrix=distances, radius=(3 + 4 * alpha) * r)

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
    # For each point x in P, compute all distances (x, S), sort, exclude z largest,
    # return largest among remaining

    dists = [min([euclidean(x, center) for center in S]) for x in P]
    dists.sort(reverse=True)
    return max(dists[z:])

    # dists = []
    # for x in P:
    #     dists.append(min([euclidean(x, center) for center in S]))
    # dists.sort(reverse=True)
    # return max(dists[z:])



def main(argv):
    file_path = argv[1]
    k = int(argv[2])
    z = int(argv[3])

    # Read points
    assert(isfile(file_path))

    inputPoints = array(readVectorsSeq(file_path))
    weights = array([1 for _ in range(len(inputPoints))])
    solution = SeqWeightedOutliers(inputPoints, weights, k, z, alpha=0)


def test():
    # args - PATH, K, Z
    main([' ', './testdataHW2.txt', '3', '3'])
    print()

    main([' ', './testdataHW2.txt', '3', '1'])
    print()

    main([' ', './testdataHW2.txt', '3', '0'])
    print()

    # main([' ', './artificial9000.txt', '9', '300'])
    # print()

    main([' ', './uber-small.csv', '10', '100'])
    print()

    main([' ', './uber-small.csv', '10', '0'])
    print()

if __name__ == '__main__':
    # test()
    main(sys.argv)



