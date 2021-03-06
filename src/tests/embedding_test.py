# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 20:47:14 2021

@author: vorst
"""

# Python imports
import unittest

# Third party imports
import numpy as np
from scipy.sparse import csr_matrix

# Local imports
from pyMILES.embedding import embed_bag, embed_all_bags, most_likely_estimator

# Globals
INSTANCE_SPACE = 2
N_POSITIVE_BAGS = 20
N_NEGATIVE_BAGS = 20

N_BAGS = 465
BAG_SIZE = 17
N_CONCEPT_CLASS_INSTANCES = 1287
P_FEATURE_SPACE = 37

# %%


class EmbeddingTest(unittest.TestCase):
    """Test methods from pyMILES.embedding"""
    
    def setUp(self):

        self.positive_bags, self.negative_bags = self.generate_dummy_data()

        # C is the concept class - the set of all training instances
        # From positive and negative bags.
        # C = {x^k : k=1, ..., n}
        # Where x^k is the kth instance in the entire training set
        self.C = np.concatenate((
            np.reshape(self.positive_bags, (BAG_SIZE *
                       N_POSITIVE_BAGS, INSTANCE_SPACE)),
            np.reshape(self.negative_bags, (BAG_SIZE*N_NEGATIVE_BAGS, INSTANCE_SPACE))),
            axis=0)

        return None

    def generate_dummy_data(self):
        # Create some dummy data
        """each instance is generated by one of the following two-dimensional 
        probability distributions: N1([5,5]^T, I), 
        N2([5,-5]^T, I), 
        N3([-5,5]^T, I),
        N4([-5,-5]^T, I), 
        N5([0,0]^T, I). N([5,5]^T, I) denotes the normal distribution with mean 
        [5,5] and identity covariance matrix"""
        n = [([5, 5], [1, 1]),
             ([5, -5], [1, 1]),
             ([-5, 5], [1, 1]),
             ([-5, -5], [1, 1]),
             ([0, 0], [1, 1]), ]

        # Create 20 positive bags, and 20 negative bags
        """A bag is labeled positive if it contains instances from at 
        least two different distributions among N1, N2, and N3"""
        positive_bags = np.zeros((N_POSITIVE_BAGS, BAG_SIZE, INSTANCE_SPACE))
        negative_bags = np.zeros((N_NEGATIVE_BAGS, BAG_SIZE, INSTANCE_SPACE))

        for i in range(0, N_POSITIVE_BAGS):

            # Fill with 2 instances from positive distribution
            distributions = np.random.randint(0, 100, BAG_SIZE)
            positive_bags[i, 0, :] = np.random.normal(n[distributions[0] % 3][0],  # Mean
                                                      # Standard Deviation
                                                      n[distributions[0] %
                                                          3][1],
                                                      INSTANCE_SPACE)  # Size
            positive_bags[i, 1, :] = np.random.normal(n[distributions[1] % 3][0],  # Mean
                                                      # Standard Deviation
                                                      n[distributions[1] %
                                                          3][1],
                                                      INSTANCE_SPACE)  # Size

            for j in range(2, BAG_SIZE):
                # Fill with instances from any other distribution
                positive_bags[i, j, :] = np.random.normal(n[distributions[j] % 5][0],  # Mean
                                                          # Standard Deviation
                                                          n[distributions[j] %
                                                              5][1],
                                                          INSTANCE_SPACE)  # Size

        for i in range(0, N_NEGATIVE_BAGS):
            # Fill with distributions, but maximum of 1 from n1,n2,n3
            distributions = np.random.randint(0, 100, BAG_SIZE)
            flag = False
            for j in range(0, BAG_SIZE):
                mod = distributions[j] % 5
                if flag:
                    # Only allow a single instance from positive distribution
                    mod = np.random.randint(3, 5)  # 3 or 4
                if mod in [0, 1, 2]:
                    flag = True
                # Fill with instances from any other distribution
                negative_bags[i, j, :] = np.random.normal(n[mod][0],  # Mean
                                                          # Standard Deviation
                                                          n[mod][1],
                                                          INSTANCE_SPACE)  # Size

        return positive_bags, negative_bags

    def test_embed_bag(self):

        concept_class = np.array(([1, 2, 3, 4, 5, 6],
                                 [-1, -2, -3, -4, -5, -6],
                                 [100, 200, 300, 400, 500, 600]))
        bag = np.array(([1, 2, 3, 4, 5, 6],
                       [7, 8, 9, 10, 11, 12],
                       [13, 14, 15, 16, 17, 18]),)
        sigma = 1  # Higher values regularize less
        distance = 'euclidean'

        embedded_bag = embed_bag(concept_class,
                                 bag,
                                 sigma,
                                 distance)
        msg = "Concept exactly matches an instance in the bag. Similarity should be 1.0"
        self.assertEqual(embedded_bag[0], 1.0, msg)
        msg = "Concept does not match instance in the bag. Similarity should be ~0"
        self.assertAlmostEqual(embedded_bag[1], 0.0, places=3, msg=msg)
        self.assertAlmostEqual(embedded_bag[2], 0.0, places=3, msg=msg)

        return None

    def test_embed_all_bags(self):

        # numpy array of shape (k,p)
        rng = np.random.default_rng()
        concept_class = rng.random((N_CONCEPT_CLASS_INSTANCES, P_FEATURE_SPACE),
                                   dtype=np.float32)

        # Numpy array of shape (i,j,p)
        bags = rng.random((N_BAGS, BAG_SIZE, P_FEATURE_SPACE),
                          dtype=np.float32)

        embedded_bags = embed_all_bags(concept_class,
                                       bags,
                                       sigma=3,
                                       distance='euclidean')

        return None

    def test_embed_all_bags_list(self):

        # numpy array of shape (k,p)
        rng = np.random.default_rng()
        concept_class = rng.random((N_CONCEPT_CLASS_INSTANCES, P_FEATURE_SPACE),
                                   dtype=np.float32)

        # Numpy array of shape (i,j,p)
        bags = []
        for n in range(0, N_BAGS):
            bag = rng.random((BAG_SIZE + rng.integers(low=-5, high=5, size=1)[0],
                              P_FEATURE_SPACE), dtype=np.float32)
            bags.append(bag)

        embedded_bags = embed_all_bags(concept_class,
                                       bags,
                                       sigma=3,
                                       distance='euclidean')

        return None

    def test_most_likely_estimator(self):

        concept = np.array([1, 2, 3, 4, 5, 6])
        bag = np.array(([1, 2, 3, 4, 5, 6],
                       [7, 8, 9, 10, 11, 12],
                       [13, 14, 15, 16, 17, 18]),)
        sigma = 3
        distance = 'euclidean'

        similarity = most_likely_estimator(concept,
                                           bag,
                                           sigma,
                                           distance)
        msg = "Concept exactly matches an instance in the bag. Similarity should be 1.0"
        self.assertEqual(similarity, 1.0, msg)

        return None

    def test_embed_all_bags_sparse(self):

        # scipy.sparse.csr_matrix of shape (k,p)
        concept_class = csr_matrix((N_CONCEPT_CLASS_INSTANCES, P_FEATURE_SPACE),
                                   dtype=np.float32)

        # Numpy array of numpy arrays, dense
        rng = np.random.default_rng()
        bags = []
        for n in range(0, N_BAGS):
            bag = np.random.rand(
                BAG_SIZE + rng.integers(low=-5, high=5, size=1)[0], P_FEATURE_SPACE)
            bags.append(csr_matrix(bag))

        with self.assertRaises(ValueError):
            embedded_bags = embed_all_bags(concept_class,
                                           bags,
                                           sigma=3,
                                           distance='euclidean')

        return None


if __name__ == '__main__':
    unittest.main()
