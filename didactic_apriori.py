import numpy as np
import itertools
from collections import defaultdict


class DidatticApriori:
    def __init__(self, min_sup, sup_type='a'):

        self.min_sup = min_sup
        self.sup_type = sup_type

        self.itemset_count = defaultdict(int)
        self.itemset_unfrequent = dict()

    def __count_itemsets(self, transactions, max_len):
        for transaction in transactions:
            for itemset in itertools.combinations(transaction, max_len):
                antomonotonic = True
                for i in range(1, max_len + 1):
                    for subitemset in itertools.combinations(itemset, i):
                        if subitemset in self.itemset_unfrequent:
                            antomonotonic = False
                if antomonotonic:
                    self.itemset_count[itemset] += 1

    def __remove_unfrequent(self):
        for itemset in self.itemset_count:
            if self.itemset_count[itemset] < self.min_sup:
                self.itemset_unfrequent[itemset] = 0

    def __print_itemsets(self, max_len, num_transactions):
        print('Apriori - Iteration %d' % max_len)
        count = 0
        count_unfrequent = 0
        for itemset in sorted(self.itemset_count):
            unfrequent = 'X' if itemset in self.itemset_unfrequent else ''
            if len(itemset) >= max_len:
                sup = self.itemset_count[itemset]
                if self.sup_type == 'r':
                    sup /= float(num_transactions)
                print(itemset, '%.2f' % sup, unfrequent)
                count += 1
                if unfrequent == 'X':
                    count_unfrequent += 1

        return count == count_unfrequent

    def fit(self, transactions, step_by_step=False):
        
        sorted_transactions = list()
        for transaction in transactions:
            sorted_transactions.append(sorted(transaction))
        transactions = sorted_transactions

        max_len = 1
        self.num_transactions = len(transactions)

        if self.sup_type == 'r':
            self.min_sup = np.round(len(transactions) * self.min_sup)

        stop = False
        while not stop:
            self.__count_itemsets(transactions, max_len)
            self.__remove_unfrequent()
            stop = self.__print_itemsets(max_len, self.num_transactions)
            max_len += 1

            if step_by_step and not stop:
                val = input('Continue')
                print('')

        return self

    def __calc_conf(self, itemset, head, tail):
        sup_itemset = self.itemset_count[itemset]
        sup_head = self.itemset_count[head]
        sup_tail = self.itemset_count[tail]
        conf = 0.0 if sup_head == 0.0 else 1.0 * sup_itemset / sup_head
        return conf

    def __calc_lift(self, itemset, head, tail):
        sup_itemset = self.itemset_count[itemset]
        sup_head = self.itemset_count[head]
        sup_tail = self.itemset_count[tail]
        num = 1.0 * sup_itemset / self.num_transactions
        den = (1.0 * sup_head / self.num_transactions) * (1.0 * sup_tail / self.num_transactions)
        lift = 0.0 if den == 0.0 else num / den
        return lift

    def __print_rules(self):

        for rule in sorted(self.rule_conf):
            unfrequent = 'X' if rule in self.rule_unfrequent else ''
            lift = 'lift: %.2f' % self.rule_lift[rule] if rule in self.rule_lift else ''
            print('%s --> %s' % (rule[0], rule[1]), 'conf: %.2f' % self.rule_conf[rule], unfrequent, lift)

    def extract_rules(self, min_conf):
        self.min_conf = min_conf
        self.rule_conf = dict()
        self.rule_unfrequent = dict()
        self.rule_lift = dict()

        analyzed_rules = set()
        for itemset in sorted(self.itemset_count):
            if itemset in self.itemset_unfrequent:
                continue
            if len(itemset) >= 2:
                for headtail in itertools.permutations(itemset):
                    for i in range(1, len(headtail)):
                        head = tuple(sorted(headtail[:i]))
                        tail = tuple(sorted(headtail[i:]))
                        rule = tuple([head, tail])
                        if rule in analyzed_rules:
                            continue
                        analyzed_rules.add(rule)    
                        self.rule_conf[rule] = self.__calc_conf(itemset, head, tail)
                        if self.rule_conf[rule] < self.min_conf:
                            self.rule_unfrequent[rule] = 0
                        else:
                            self.rule_lift[rule] = self.__calc_lift(itemset, head, tail)

        self.__print_rules()

