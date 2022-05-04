import time
import sys
from sklearn.metrics import pairwise_distances
from os.path import isfile

from numpy import min
from numpy import copy
from numpy import arange
import numpy as np
from numpy import sum
from numpy import array

from math import sqrt
from queue import LifoQueue


def readVectorsSeq(filename):
    with open(filename) as f:
        result = [tuple(map(float, i.split(','))) for i in f]
    return result


def euclidean(point1, point2):
    res = 0
    for i in range(len(point1)):
        diff = (point1[i] - point2[i])
        res += diff * diff
    return sqrt(res)

def calc_ball_weight(indices, weights, idx, radius, distance_matrix):
    # Optimized list comprehension
    # return sum([weights[i] for i in indices if (i != idx) and (distance_matrix[idx][i] <= radius)])

    # More understandable code
    sum_weight = 0
    for i in indices:
        if (i != idx) and (distance_matrix[idx][i] <= radius):
            sum_weight += weights[i]
    return sum_weight


def get_ball_indices(idxs, distance_matrix, x_idx, radius):
    res = []
    for i in idxs:
        # print(f'Dist: {distance_matrix[x_idx][i]}, radius: {radius}')
        if distance_matrix[x_idx][i] <= radius and x_idx != i:
            res.append(i)
    return res

def find_new_center(pointset, Z_idxs, weights, radius, distances, center_indices):
    max_weight = 0
    new_center_idx = None
    for i in range(len(pointset)):
        if not (i in center_indices):
            ball_weight = calc_ball_weight(
                          Z_idxs, weights, i, radius, distances
            )
            ball_weight += weights[i]  # Account for the weight of x

            if ball_weight > max_weight:
                max_weight = ball_weight
                new_center_idx = i
    print(f'Max weight: {max_weight}')
    return new_center_idx


def SeqWeightedOutliers(P, W, k, z, alpha):

    distances = pairwise_distances(P)

    subset = P[: k + z + 1]
    r_dist = pairwise_distances(subset)
    r = np.min(r_dist[r_dist != 0]) / 2

    print(f'Initial guess r = {r}')

    while(True):
        Z_idxs = [i for i in range(len(P))]
        S_idxs = []
        Wz = sum([w for w in W])

        while (len(S_idxs) < k) and (Wz > 0):
            # finding x in P with max weight of the ball
            new_center_idx = find_new_center(
                pointset=P,
                Z_idxs=Z_idxs,
                weights=W,
                radius=(1 + 2 * alpha) * r,
                distances=distances,
                center_indices=S_idxs # we search for all x from P, but if x is not in already chosen set of centers
            )
            print(f'New center index: {new_center_idx}')
            assert(not (new_center_idx is None))
            assert(not (new_center_idx in S_idxs))


            S_idxs.append(new_center_idx)

            # Bz(newcenter, (3 + 4alpha)r)
            Bz = get_ball_indices(Z_idxs, distances, new_center_idx, (3 + 4 * alpha) * r)
            print(f'Ball indices: {Bz}, Z indices: {Z_idxs}')

            for idx in Bz: # + [new_center_idx]:
                print(idx)
                Z_idxs.remove(idx)
                Wz -= W[idx]

        if (Wz <= z):
            print(f'final r = {r}')
            return P[S_idxs]
        else:
            r *= 2






def ComputeObjective(P, S, z):
    # For each point x in P, compute all distances (x, S), sort, exclude z largest,
    # return largest among remaining
    dists = []
    for x in P:
        for center in S:
            dists.append(euclidean(x, center))
    dists.sort(reverse = True)
    return max(dists[z : ])




def main(argv):
    file_path = argv[1]
    k = int(argv[2])
    z = int(argv[3])

    # Read points
    assert(isfile(file_path))
    inputPoints = array(readVectorsSeq(file_path))

    # print(inputPoints)

    # Create weights
    weights = array([1 for _ in range(len(inputPoints))])
    # print(weights)

    # Run SeqWeightedOutliers(inputPoints, weights, k, z, 0)
    # compute a set of at most k centers



    solution = SeqWeightedOutliers(inputPoints, weights, k, z, 0)
    print(solution)
    objective = ComputeObjective(inputPoints, solution, z)
    print(f'Objective function = {objective}')


if __name__ == '__main__':

    # main(sys.argv)
    path = './testdataHW2.txt'
    k = '3'
    z = '3'
    args = [' ', path, k, z]
    main(args)


