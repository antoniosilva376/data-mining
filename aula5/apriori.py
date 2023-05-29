from collections import Counter
from itertools import chain, combinations

class TransactionDataset:
    def __init__(self, transactions):
        self.transactions = transactions

    def get_transactions(self):
        return self.transactions

class Apriori:
    def __init__(self, dataset, min_support):
        self.dataset = dataset
        self.min_support = min_support

    def generate_frequent_itemsets(self):
        transactions = self.dataset.get_transactions()

        subsets = list(
            chain.from_iterable(combinations(subset, r) for subset in transactions for r in range(1, len(subset) + 1)))
        counts = Counter([''.join(sorted(subset)) for subset in subsets])
        #print(counts)
        """
        itemsets = {}
        for transaction in transactions:
            for item in counts:
                if item not in itemsets:
                    itemsets[item] = 1
                else:
                    itemsets[item] += 1
        print(itemsets)"""
        frequent_itemsets = []
        for itemset in counts:
            if counts[itemset] >= self.min_support:
                frequent_itemsets.append(list(itemset))
        return frequent_itemsets

    def generate_association_rules(self, min_confidence):
        frequent_itemsets = self.generate_frequent_itemsets()
        rules = []
        print("frequent_itemsets", frequent_itemsets)
        for itemset in frequent_itemsets:
            if len(itemset) > 1:
                for item in itemset:
                    consequent = [item]
                    antecedent = [x for x in itemset if x != item]
                    confidence = self.calculate_confidence(antecedent, consequent)
                    if confidence >= min_confidence:
                        rules.append((antecedent, consequent, confidence))
        rules.sort()
        return rules

    def calculate_confidence(self, antecedent, consequent):
        transactions = self.dataset.get_transactions()
        antecedent_count = 0
        consequent_count = 0
        for transaction in transactions:
            if set(antecedent).issubset(set(transaction)):
                antecedent_count += 1
                if set(consequent).issubset(set(transaction)):
                    consequent_count += 1
        return float(consequent_count) / float(antecedent_count)

import unittest

class TestApriori(unittest.TestCase):
    def test_generate_frequent_itemsets(self):
        transactions = [['A', 'B', 'C'], ['A', 'B'], ['A', 'C'], ['B', 'C']]
        dataset = TransactionDataset(transactions)
        apriori = Apriori(dataset, 2)
        frequent_itemsets = apriori.generate_frequent_itemsets()
        self.assertEqual(frequent_itemsets, [['A'], ['B'], ['C'], ['A', 'B'], ['A', 'C'], ['B', 'C']])

    def test_generate_association_rules(self):
        transactions = [['A', 'B', 'C'], ['A', 'B'], ['A', 'C'], ['B', 'C']]
        dataset = TransactionDataset(transactions)
        apriori = Apriori(dataset, 2)
        rules = apriori.generate_association_rules(0.5)
        expected = [(['A'], ['B'], 0.6666666666666666), (['B'], ['A'], 0.6666666666666666), (['A'], ['C'], 0.6666666666666666), (['C'], ['A'], 0.6666666666666666), (['B'], ['C'], 0.6666666666666666), (['C'], ['B'], 0.6666666666666666)]
        expected.sort()
        self.assertEqual(rules, expected)


if __name__ == '__main__':
    unittest.main()
