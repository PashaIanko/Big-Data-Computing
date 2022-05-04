import time
import sys
from sklearn.metrics import pairwise_distances
from os.path import isfile

from numpy import min
from numpy import copy
from numpy import arange
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

def calc_ball_weight(pointset, weights, idx, radius, distance_matrix):
    # Optimized list comprehension
    return sum([weights[i] for i in range(len(pointset)) if (i != idx) and (distance_matrix[idx][i] <= radius)])

    # More understandable code
    sum_weight = 0
    # for i, point in enumerate(pointset):
    #     if (i != idx) and (distance_matrix[idx][i] <= radius):
    #         sum_weight += weights[i]
    # return sum_weight


def get_ball_indices(idxs, distance_matrix, x_idx, radius):
    res = []
    for i in idxs:
        # print(f'Dist: {distance_matrix[x_idx][i]}, radius: {radius}')
        if distance_matrix[x_idx][i] <= radius and x_idx != i:
            res.append(i)
    return res

    # return [i for i in idxs if distance_matrix[x_idx][i] <= radius and x_idx != i]



def SeqWeightedOutliers(P, W, k, z, alpha):
    # Weighted variant of kcenterOUT
    # P - pointset
    # W - weights
    # k - number of centers
    # z - number of outlierz
    # alpha - euristics coefficient

    distances = pairwise_distances(P)
    r = min(distances[distances != 0][: k + z + 1]) / 2
    print(f'Initial r: {r}')

    while True:
        Z_idxs = [i for i in range(len(P))]  # arange(len(P))
        S_idxs = []
        Wz = sum(W)
        iter = 0
        while (len(S_idxs) < k) and (Wz > 0):
            max_weight = 0
            newcenter_idx = None

            print(f'Iteration {iter}: Remain {len(Z_idxs)} in Z. Len P = {len(P)}')
            iter += 1
            for i, _ in enumerate(P):

                Bz_idxs = get_ball_indices(
                    idxs=Z_idxs,  # allowed indexes - which points of P are still in Z
                    distance_matrix=distances,
                    x_idx=i,
                    radius=(1 + 2 * alpha) * r
                )
                print(f'Found {len(Bz_idxs)} objects in Bz')
                ball_weight = sum(W[Bz_idxs]) + W[i]  # Weight of the ball center is also counted
                print(f'Ball weight = {ball_weight}')


                if ball_weight > max_weight and not (newcenter_idx in S_idxs):
                    max_weight = ball_weight
                    print(f'Updated Now max = {max_weight}, ball weight = {ball_weight}')
                    newcenter_idx = i



            S_idxs.append(newcenter_idx)

            Bz_ = get_ball_indices(
                idxs=Z_idxs,
                distance_matrix=distances,
                x_idx=newcenter_idx,
                radius=(3 + 4 * alpha) * r
            )

            for idx in Bz_ + [newcenter_idx]: # After adding a center to C, we also remove the center from Z
                print(f'Idx: {idx}')
                if idx in Z_idxs:
                    Z_idxs.remove(idx)
                Wz -= W[idx]
        if Wz <= z:
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
    print(objective)


if __name__ == '__main__':

    # main(sys.argv)
    path = './testdataHW2.txt'
    k = '3'
    z = '3'
    args = [' ', path, k, z]
    main(args)


