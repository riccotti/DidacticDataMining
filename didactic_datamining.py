from didactic_kmeans import *
from didactic_apriori import *
from didactic_classificationtree import *
from didactic_dbscan import *
from didactic_hierarchical import *


def create_dataset(npoints=10, minvalue=0, maxvalue=10):
    while True:
        dataset = np.random.randint(minvalue, maxvalue, size=(npoints, 2))
        exit_flag = True
        for i in range(0, len(dataset) - 1):
            for j in range(i + 1, len(dataset)):
                if np.array_equal(dataset[i], dataset[j]):
                    exit_flag = False
                    break

            if not exit_flag:
                break

        if exit_flag:
            break

    return dataset


def print_dataset(dataset):
    for i, p in enumerate(dataset):
        print('P%d' % i, p)


def create_transactional_dataset(num_transaction=10, num_items=6, min_len=2, max_len=4):
    items = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
             'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    transactions = list()
    for i in range(0, num_transaction):
        transaction = np.random.choice(items[:num_items],
                                       np.random.randint(min_len, max_len + 1), replace=False)
        transactions.append(sorted(transaction))

    return transactions


def print_transactions(transactions):
    for transaction in transactions:
        print(transaction)

