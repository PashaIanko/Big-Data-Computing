import time
import sys
from math import sqrt


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


def main(argv):
    pass

if __name__ == 'main':

    # main(sys.argv)
    path = './testdataHW2.txt'
    k = 3
    z = 3
    args = [' ', path, chr(k), chr(z)]
    main(args)


