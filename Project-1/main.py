import os
import sys

from pyspark import SparkContext
from pyspark import SparkConf
import random as rand


def check_input(args):
    assert len(args) == 4, 'Incorrect № parameters'

    k = argv[0]
    h = argv[1]
    _ = argv[2]
    data_path = argv[3]

    assert k.isdigit(), 'Incorrect K (expect number)'
    assert h.isdigit(), 'Incorrect H (expect number)'
    assert os.path.isfile(data_path), 'Cannot access file, check directory'

def filter_data(RDD, country):
    filtered = RDD.filter(
        lambda item: int(item[3]) > 0
    )

    if country == 'all':
        return filtered
    else:
        return filtered.filter(
            lambda item: item[-1] == country
        )

def gather_unique_customers(pairs):
    prod_preference = {}

    for index, pair in pairs:
        product = pair[0]
        customer = pair[1]
        if product not in prod_preference.keys():
            prod_preference[product] = 1
        else:
            prod_preference[product] += 1
    return list(prod_preference.items())

def print_output(popularity_items):
    sort = sorted(popularity_items, key=lambda x: int(x[0]))
    for item in sort:
        print(f'Product: {item[0]}, popularity: {item[1]}')

def print_ordered_pairs(items_list):
    for item in sorted(items_list, key=lambda item: int(item[0])):
        print(item)

def print_top(productRDD, top_n):
    top_list = productRDD.top(top_n, key=lambda x: x[1])
    for item in top_list:
        print(f'Product ID: {item[0]}, Popularity: {item[1]}')

def main(argv):
    # check correctness
    check_input(argv)

    k = int(argv[0])
    h = int(argv[1])
    s = argv[2]
    data_path = argv[3]

    # set up spark session
    conf = SparkConf().setAppName('CustomerService').setMaster("local[*]")
    sc = SparkContext(conf=conf)
    sc.setLogLevel('ERROR')

    # 1. read into rawData
    rawData = sc.textFile(data_path, minPartitions=k)
    rawData.repartition(numPartitions=k)
    print(f'Number of rows = {rawData.count()}')

    # 2. productCustomer
    # first, to keep space safe, we will filter data. filter returns
    # new RDD
    rawData = rawData.map(lambda doc: doc.split(','))
    rawData = filter_data(rawData, s)

    # Map into pairs product ID -- customer ID.
    # We will be reducing by key, so we dont need any value
    # in the key-val pair --> we substitute it with default 0
    rawData = rawData.map(lambda item: ((f'{item[1]}, {item[6]}'), 0))

    # Distinct pairs of product -- customer
    productCustomer = rawData.reduceByKey(lambda x, y: x + y).map(
        lambda item: (item[0].split(',')[0], int(item[0].split(',')[1]))
    )
    print(f'Product-Customer Pairs = {productCustomer.count()}')

    # 3. Product popularity with mapPartitionsToPair / mapPartitions
    # Make partitions
    productPopularity1 = productCustomer.map(lambda item: (rand.randint(0, k - 1), item))
    productPopularity1 = productPopularity1.mapPartitions(gather_unique_customers)
    productPopularity1 = productPopularity1.groupByKey()
    productPopularity1 = productPopularity1.mapValues(lambda vals: sum(vals))
    print(f'productPopularity1:')
    print_output(productPopularity1.collect())
    print('\n')

    # 4. Product popularity with map / mapToPair / reduceByKey methods
    productPopularity2 = productCustomer\
            .map(lambda item: (item[0], 1))\
            .reduceByKey(lambda x, y: x + y)
    print(f'productPopularity2:')
    print_output(productPopularity2.collect())
    print('\n')

    # 5. Extract top h values
    if h > 0:
        print(f'Top {h} popular products:\n')
        print_top(productPopularity1, h)

    # 6. If h == 0
    if h == 0:
        print(f'Ordered pairs for popularity 1:\n')
        print_ordered_pairs(productPopularity1.collect())
        print(f'Ordered pairs for popularity 2:\n')
        print_ordered_pairs(productPopularity2.collect())


if __name__ == "__main__":
    os.environ['pyspark_python'] = sys.executable
    os.environ['pyspark_driver_python'] = sys.executable

    # K, h, s, data_path
    argv = ['4', '2', 'Italy', './sample_50.csv']

    main(argv)
    # main(sys.argv)