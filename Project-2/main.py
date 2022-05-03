import time
import sys
from sklearn.metrics import pairwise_distances
from os.path import isfile

from numpy import min
from numpy import copy
from numpy import sum

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





def SeqWeightedOutliers(P, W, k, z, alpha):
    # Weighted variant of kcenterOUT
    # P - pointset
    # W - weights
    # k - number of centers
    # z - number of outlierz
    # alpha - euristics coefficient

    distances = pairwise_distances(P)
    r = min(distances[distances != 0][: k + z + 1])

    while True:
        Z = copy(P)
        S = LifoQueue() # []  # classes
        Wz = sum(W)
        while (len(S) < k) and (Wz > 0):
            max = 0
            newcenter = None
            for i, x in enumerate(P):
                ball_weight = calc_ball_weight(
                    pointset=Z,
                    weights=W,
                    idx=i,
                    radius=(1 + 2 * alpha) * r,
                    distance_matrix = distances
                )
                if ball_weight > max:
                    max = ball_weight
                    newcenter = x
            S.put(newcenter)





        del Z



def ComputeObjective(P, S, z):
    pass


def main(argv):
    file_path = argv[1]
    k = argv[2]
    z = argv[3]

    # Read points
    assert(isfile(file_path))
    inputPoints = readVectorsSeq(file_path)

    # print(inputPoints)

    # Create weights
    weights = [1 for _ in range(len(inputPoints))]
    # print(weights)

    # Run SeqWeightedOutliers(inputPoints, weights, k, z, 0)
    # compute a set of at most k centers



    solution = SeqWeightedOutliers(inputPoints, weights, k, z, 0)
    # objective = ComputeObjective(inputPoints, solution, z)


if __name__ == '__main__':

    # main(sys.argv)
    path = './testdataHW2.txt'
    k = 3
    z = 3
    args = [' ', path, chr(k), chr(z)]
    main(args)


