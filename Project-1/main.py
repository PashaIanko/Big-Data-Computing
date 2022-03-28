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
    # TODO: filtering Rovena
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

def do_partition(client_log, n_partitions):
    return (rand.randint(0, n_partitions - 1), client_log)

def gather_unique_pairs(pairs, country):
    prod_customer_dict = {}

    rand_key, client_logs = pairs[0], pairs[1]

    for log in client_logs:

        features = log.split(',')
        quantity = features[3]
        if (int(quantity) > 0) and (country == 'all' or country == features[-1]):  # quantity
            product = features[1]
            customer = features[6]

            if product in prod_customer_dict.keys():
               if not (customer in prod_customer_dict[product]):
                   prod_customer_dict[product].append(customer)
            else:
               prod_customer_dict[product] = [customer]

    res = []
    for productID, customers in prod_customer_dict.items():
        for customer in customers:
            res.append((productID, customer))
    return res

def calc_popularity(pairs):
    popularity_dict = {}

    for pair in pairs:
        product, customer = pair[0], pair[1]
        if product in popularity_dict.keys():
            if customer not in popularity_dict[product]:
                popularity_dict[product].append(customer)
        else:
            popularity_dict[product] = [customer]

    return [(product, len(customers)) for product, customers in popularity_dict.items()]

def count_popularity(pair):
    rand_idx, prod_customer_pairs = pair[0], pair[1]
    return calc_popularity(prod_customer_pairs)


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
    # Logic:
    # 1. Random partitioning into K groups
    # 2. Within each group, split item and do filtering before looking for distinct pairs
    # 3. Within each group, look for distinct product - customer pairs --> return them
    # 4. We dont need reduce by key here

    # TODO: map() Rovena
    productCustomer = rawData\
        .map(lambda client_log: do_partition(client_log, k))\
        .groupByKey()\
        .flatMap(lambda group: gather_unique_pairs(group, s))
    print(f'Product-Customer Pairs = {productCustomer.count()}')

    # 3. Product popularity:
    # 1. MapPartitions: calc number of unique customers, buying the product, group by
    # key and calc sum
    productPopularity1 = productCustomer\
        .mapPartitions(lambda x: x)\
        .groupByKey()\
        .mapValues(lambda l: len(l))

        # .mapPartitions(lambda pairs: calc_popularity(pairs))\
        # .groupByKey()\
        # .mapValues(lambda n_customers: sum(n_customers))

    print(f'productPopularity1:')
    print_output(productPopularity1.collect())
    print('\n')

    # 4. Product popularity with map / mapToPair / reduceByKey methods
    # Logic:
    # 1. Random partitioning 0 .. N-1
    # 2. Group by random key
    # 3. flatMap with products popularity
    # 4. reduceByKey
    productPopularity2 = productCustomer\
        .map(lambda item: (rand.randint(0, k - 1), item))\
        .groupByKey()\
        .flatMap(count_popularity)\
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
