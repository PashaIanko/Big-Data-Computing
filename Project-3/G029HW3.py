# Import Packages
from pyspark import SparkConf, SparkContext
import numpy as np
from numpy import array
import time
import random
import sys
import math
# from timeit import default_timer
from sklearn.metrics import pairwise_distances
from os.path import isfile
from os import environ

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# AUXILIARY METHODS
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method strToVector: input reading
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def strToVector(str):
    out = tuple(map(float, str.split(',')))
    return out



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method squaredEuclidean: squared euclidean distance
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def squaredEuclidean(point1,point2):
    res = 0
    for i in range(len(point1)):
        diff = (point1[i]-point2[i])
        res +=  diff*diff
    return res



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method euclidean:  euclidean distance
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def euclidean(point1,point2):
    res = 0
    for i in range(len(point1)):
        diff = (point1[i]-point2[i])
        res +=  diff*diff
    return math.sqrt(res)



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method MR_kCenterOutliers: MR algorithm for k-center with outliers
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def MR_kCenterOutliers(points, k, z, L):

    
    #------------- ROUND 1 ---------------------------
    # Partition into L coresets
    round_1_start = time.time()  # default_timer()
    coreset = points.mapPartitions(lambda iterator: extractCoreset(iterator, k+z+1))
    round_1_end = time.time()  # default_timer()
    
    # END OF ROUND 1

    
    #------------- ROUND 2 ---------------------------
    round_2_start = time.time()  # default_timer()

    elems = coreset.collect()
    coresetPoints = list()
    coresetWeights = list()
    for i in elems:
        coresetPoints.append(i[0])
        coresetWeights.append(i[1])

    # ****** ADD YOUR CODE
    # ****** Compute the final solution (run SeqWeightedOutliers with alpha=2)
    # ****** Measure and print times taken by Round 1 and Round 2, separately
    # ****** Return the final solution

    centers = SeqWeightedOutliers(
        P=array(coresetPoints),
        W=array(coresetWeights),
        k=k,
        z=z,
        alpha=2
    )

    round_2_end = time.time()  # default_timer()

    print(f'Time Round 1: {(round_1_end - round_1_start) * 1000} ms')
    print(f'Time Round 2: {(round_2_end - round_2_start) * 1000} ms')

    return centers



   

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method extractCoreset: extract a coreset from a given iterator
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def extractCoreset(iter, points):
    partition = list(iter)
    centers = kCenterFFT(partition, points)
    weights = computeWeights(partition, centers)
    c_w = list()
    for i in range(0, len(centers)):
        entry = (centers[i], weights[i])
        c_w.append(entry)
    # return weighted coreset
    return c_w
    
    
    
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method kCenterFFT: Farthest-First Traversal
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def kCenterFFT(points, k):
    idx_rnd = random.randint(0, len(points)-1)
    centers = [points[idx_rnd]]
    related_center_idx = [idx_rnd for i in range(len(points))]
    dist_near_center = [squaredEuclidean(points[i], centers[0]) for i in range(len(points))]

    for i in range(k-1):
        new_center_idx = max(enumerate(dist_near_center), key=lambda x: x[1])[0] # argmax operation
        centers.append(points[new_center_idx])
        for j in range(len(points)):
            if j != new_center_idx:
                dist = squaredEuclidean(points[j], centers[-1])
                if dist < dist_near_center[j]:
                    dist_near_center[j] = dist
                    related_center_idx[j] = new_center_idx
            else:
                dist_near_center[j] = 0
                related_center_idx[j] = new_center_idx
    return centers



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method computeWeights: compute weights of coreset points
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def computeWeights(points, centers):
    weights = np.zeros(len(centers))
    for point in points:
        mycenter = 0
        mindist = squaredEuclidean(point,centers[0])
        for i in range(1, len(centers)):
            dist = squaredEuclidean(point,centers[i])
            if dist < mindist:
                mindist = dist
                mycenter = i
        weights[mycenter] = weights[mycenter] + 1
    return weights


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method SeqWeightedOutliers: sequential k-center with outliers
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# def SeqWeightedOutliers (points, weights, k, z, alpha):
# #
# # ****** ADD THE CODE FOR SeqWeightedOuliers from HW2
# #
#     pass

def calc_ball_weight(idx, z_idxs, weights, distance_matrix, radius):
    return sum([weights[z_index] for z_index in z_idxs if distance_matrix[z_index][idx] <= radius])

def find_new_center_idx(input_size, Z_idxs, weights, distance_matrix, radius):
    ball_weights = [calc_ball_weight(idx=i, z_idxs=Z_idxs, weights=weights, distance_matrix=distance_matrix, radius=radius) for i in range(input_size)]
    return ball_weights.index(max(ball_weights))

def find_ball_indices(idx, idxs, distance_matrix, radius):
    return [z_idx for z_idx in idxs if distance_matrix[idx][z_idx] <= radius]


def SeqWeightedOutliers(P, W, k, z, alpha):
    # start = default_timer()

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
            # end = default_timer()

            # print(f'Input size n = {input_size}')
            # print(f'Number of centers k = {k}')
            # print(f'Number of outliers z = {z}')
            print(f'Initial guess = {r_initial}')
            print(f'Final guess = {r}')
            print(f'Number of guesses = {n_guesses}')
            # print(f'Objective function = {ComputeObjective(P, P[S_idxs], len(Z_idxs))}')
            # print(f'Time of SeqWeightedOutliers = {(end - start) * 1000}')

            return P[S_idxs]
        else:
            r *= 2
            n_guesses += 1



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method computeObjective: computes objective function
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def computeObjective(points, centers, z):
#
# ****** ADD THE CODE FOR SeqWeightedOuliers from HW2
#
    # Here we have to account, that InputPoints is
    # a large pointset


    dists = [min([euclidean(x, center) for center in centers]) for x in points.collect()]
    dists.sort(reverse=True)
    return max(dists[z:])



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# MAIN PROGRAM
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def main(argv):
    # Checking number of cmd line parameters
    assert len(argv) == 5, "Usage: python Homework3.py filepath k z L"

    # Initialize variables
    filename = argv[1]
    k = int(argv[2])
    z = int(argv[3])
    L = int(argv[4])

    start = 0
    end = 0

    # Set Spark Configuration
    conf = SparkConf().setAppName('MR k-center with outliers')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")


    assert isfile(filename), print('Cannot find file')
    # Read points from file

    start = time.time()
    inputPoints = sc.textFile(filename, L).map(lambda x: strToVector(x)).repartition(L).cache()
    N = inputPoints.count()
    end = time.time()

    # Pring input parameters
    print("File : " + filename)
    print("Number of points N = ", N)
    print("Number of centers k = ", k)
    print("Number of outliers z = ", z)
    print("Number of partitions L = ", L)
    print("Time to read from file: ", str((end - start) * 1000), " ms")

    # Solve the problem
    solution = MR_kCenterOutliers(inputPoints, k, z, L)

    # Compute the value of the objective function
    start = time.time()
    objective = computeObjective(inputPoints, solution, z)
    end = time.time()

    print("Objective function = ", objective)
    print("Time to compute objective function: ", str((end - start) * 1000), " ms")

def test():
    # Filepath, centers, outliers, partitions

    main([' ', './testdataHW3.txt', '3', '3', '1'])
    print()

    # main([' ', '/testdataHW2.txt', '3', '1', '1'])
    # print()

    # main([' ', '/testdataHW2.txt', '3', '0', '1'])
    # print()



    # main([' ', './artificial9000.txt', '9', '300'])
    # print()

    # main([' ', './uber-small.csv', '10', '100'])
    # print()

    # main([' ', './uber-small.csv', '10', '0'])
    # print()


# Just start the main program
if __name__ == "__main__":
    environ['pyspark_python'] = sys.executable
    environ['pyspark_driver_python'] = sys.executable

    test()
    # main(sys.argv)

