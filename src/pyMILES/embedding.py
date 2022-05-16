# -*- coding: utf-8 -*-
"""
Created on Sat May 29 18:11:55 2021

Definitions of variables and vocabulary
Vocabulary and variables attempt to match the original paper

Concept class: an array of shape (k,p) where k is the number of 
    instances in the concept class. The total number of instances is equal
    to the total number of 'concepts'. A bag contains multiple concepts.
    If each bag contains the same number of instances, then 
    k = n_bags * n_instances_per_bag
    
Bag: Text

Euclidean distance: The distance between two vectors

RBF (radial basis function) distance: This is basically the exponential of the 
    euclidean distance

Gaussian Distance: A function of the form a * exponential( -(x-b)**2 / (2*c**2))
    It is basically the exponentail of the squarred euclidean distance

Most likely estimator: The most likely estimator is a similarity measure 
    between the entire concept class C and a bag B
    C = {x^k : k=1, ..., n_instances}
    B = {x_j : j=1, ..., bag_size}
    Note: this is the estimator that maximizes the diverse density
    of a concept DD(t)

embedding (of a single bag): (np.ndarray) size (k,1) where k is the number of 
    concepts in the concept class. The kth feature in the concept class 
    realizes the kth row of the matrix. This verbiage represents the 
    embedding of a SINGLE bag
    
embedding (of all bags): (np.ndarray) size (k, i) where k is the number of 
    concepts in the concept class, and i is the total number of bags being 
    embedded. Each column in the array represents a bag,
    and the kth feature in the concept class realizes the kth row of the 
    matrix.

concept_class: (np.ndarray) of shape (k,p) where k is the number of 
    instances in the concept class, and p is the feature space of an instance. 
    The total number of instances is equal
    to the total number of 'concepts'. A bag contains multiple concepts.
    If each bag contains the same number of instances, then 
    k = n_bags * n_instances_per_bag.  The number of instances per bag can vary
    
bags: (np.ndarray) shape (i,j,p) where n is the number of bags, j is the 
    number of instances per bag, and p is the feature space per instance 
    in a bag
    
sigma: (float) Scaling factor, try values in the range (0 -> 5)
 
@author: vorst
"""

# Python imports
from typing import Union

# Third party imports
import numpy as np

# Local imports

# %%


def euclidean_distance(v1: np.ndarray,
                       v2: np.ndarray = 0) -> float:
    """Eucildean distance 
    $|v1 - v2|^2$
    inputs
    -------
    v1, v2: (np.ndarray)
    outputs
    -------
    distance: (float) distance between two vectors

    """
    return np.sqrt(np.sum((v1 - v2)**2))


def gaussian_distance(v1: np.ndarray,
                      v2: np.ndarray,
                      gamma: float = 1) -> float:
    """
    Gaussian distance function
    """
    return np.exp(-gamma * euclidean_distance(v1, v2)**2)


def radial_basis_function_distance(v1: np.ndarray,
                                   v2: np.ndarray,
                                   gamma: float = 1) -> float:
    r"""
    The Gaussian RBF kernel 
    $exp (\gama ||v1 - v2||^2)$.
    """

    return np.exp(-gamma * euclidean_distance(v1, v2))


def most_likely_estimator(concept: np.ndarray,
                          bag: np.ndarray,
                          sigma: float,
                          distance: str = 'euclidean') -> float:
    """The most likely estimator is a similarity measure between the 
    entire concept class C and a bag B
    C = {x^k : k=1, ..., n}
    B = {x_j : j=1, ..., bag_size}
    Note: this is the estimator that maximizes the diverse density
    of a concept DD(t)

    This most likeley estimator is the maximum exponential
    of the norm between a bag instance and target concept
    inputs
    -------
    concept: (np.ndarray) of shape (1,p) where p is the number of instances
        in the concept class
    bag: (np.ndarray) of shape (j,p) where j is the number of instances in 
        a bag, and p is the instance space of a bag
    sigma: (float) regularization paramter
    """

    if distance == 'euclidean':
        similarity = np.exp(
            -1 * np.square(
                np.linalg.norm(bag - concept, axis=1)) /
            (sigma ** 2)
        )
    elif distance == 'rbf':
        raise NotImplementedError(
            "Other distance functions are not implemented")
    elif distance == 'rbf2':
        raise NotImplementedError(
            "Other distance functions are not implemented")
    elif distance == 'gaussian':
        raise NotImplementedError(
            "Other distance functions are not implemented")
    else:
        raise NotImplementedError(
            "Other distance functions are not implemented")

    return max(similarity)


def embed_bag(concept_class: np.ndarray,
              bag: np.ndarray,
              sigma: float,
              distance: str = 'euclidean') -> np.ndarray:
    """Embed a bag onto a concept class (A concept class is a set of instances)
    inputs
    -------
    concept_class: (np.ndarray)
    bag: (np.ndarray) of shape (j,p) where j is the number of instances in 
        a bag, and p is the instance space of a bag
    sigma: (float) regularization paramter
    distance: (str) 'euclidean' is the only supported distance metric
    outputs
    -------
    embedding: (np.ndarray) size (k,1) where k is the number of concepts in
        the concept class. Each column represents a bag, and the kth feature 
        in the concept class realizes the kth row of the matrix
        """

    embedded_bag = np.zeros((concept_class.shape[0]))

    for n in range(0, embedded_bag.shape[0]):
        # Calculate similarity measure for each training instance
        embedded_bag[n] = most_likely_estimator(concept_class[n],
                                                bag,
                                                sigma,
                                                distance)

    return embedded_bag


def embed_all_bags(concept_class: np.ndarray,
                   bags: Union[np.ndarray, list],
                   sigma: float,
                   distance: str = 'euclidean') -> np.ndarray:
    """Embed a set of bags onto a concept class (A concept class is a set of 
    instances)
    inputs
    -------
    concept_class: (np.ndarray) of shape (k,p) where k is the number of 
        instances in the concept class. The total number of instances is equal
        to the total number of 'concepts'. A bag contains multiple concepts.
        If each bag contains the same number of instances, then 
        k = n_bags * n_instances_per_bag
    bags: (np.ndarray) shape (i,j,p) where n is the number of bags, j is the 
        number of instances per bag, and p is the feature space per instance 
        in a bag
        (list) of numpy arrays, shape (j,P)
    sigma: (float) Scaling factor, try values in the range (0 -> 5). 
        Low values greatly penalize the dissimilarity between the concept class 
        and bag.
        High values allow tolerance between the concept class and bags
    outputs
    -------
    embedding: (np.ndarray) size (k,l) where k is the number of concepts in
        the concept class, and l is the number of bags being embedded. Each column 
        represents a bag, and the kth feature in the concept class realizes the kth 
        row of the matrix
    """

    if not concept_class.ndim == 2:
        msg = ("Expecting array of shape (k,p) with 2 dimensions. " +
               "Got {} dimensions")
        raise ValueError(msg.format(concept_class.ndim))

    if hasattr(bags, 'ndim'):
        if not bags.ndim == 3:
            msg = ("Expecting bags array of shape (i,j,p) with 3 dimensions. " +
                   "Got {} dimensions")
            raise ValueError(msg.format(bags.ndim))

    if not isinstance(concept_class, np.ndarray):
        msg = ("Expecting dense numpy array. Got {}".format(type(concept_class)))
        raise ValueError(msg)

    # Embed all bags using all training instances
    if hasattr(bags, 'shape'):
        embedded_bags = np.zeros((concept_class.shape[0], bags.shape[0]))
    # List of bags
    elif hasattr(bags, '__len__'):
        embedded_bags = np.zeros((concept_class.shape[0], len(bags)))
    # Iterate through list or slices of numpy array
    for i, bag in enumerate(bags):
        embedded_bags[:, i] = embed_bag(concept_class, bag, sigma, distance)

    return embedded_bags


# %% Testing

def generate_dummy_data(bag_size: int,
                        n_positive_bags: int,
                        n_negative_bags: int) -> tuple([np.ndarray, np.ndarray]):
    """A bag is labeled positive if it contains instances from at 
    least two different distributions among N1, N2, and N3"""

    print("A bag is labeled positive if it contains instances from at "
          "least two different distributions among N1, N2, and N3")
    n = [([5, 5], [1, 1]),
         ([5, -5], [1, 1]),
         ([-5, 5], [1, 1]),
         ([-5, -5], [1, 1]),
         ([0, 0], [1, 1]), ]

    BAG_SIZE = bag_size
    INSTANCE_SPACE = 2
    N_POSITIVE_BAGS = n_positive_bags
    positive_bags = np.zeros((N_POSITIVE_BAGS, BAG_SIZE, INSTANCE_SPACE))
    N_NEGATIVE_BAGS = n_negative_bags
    negative_bags = np.zeros((N_NEGATIVE_BAGS, BAG_SIZE, INSTANCE_SPACE))

    for i in range(0, N_POSITIVE_BAGS):

        # Fill with 2 instances from positive distribution
        distributions = np.random.randint(0, 100, BAG_SIZE)
        positive_bags[i, 0, :] = np.random.normal(n[distributions[0] % 3][0],  # Mean
                                                  # Standard Deviation
                                                  n[distributions[0] % 3][1],
                                                  INSTANCE_SPACE)  # Size
        positive_bags[i, 1, :] = np.random.normal(n[distributions[1] % 3][0],  # Mean
                                                  # Standard Deviation
                                                  n[distributions[1] % 3][1],
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
