# -*- coding: utf-8 -*-
"""
Created on Sat May 29 18:11:55 2021

@author: vorst
"""

# Python imports

# Third party imports
import numpy as np

# Local imports

#%%

def euclidean_distance(v1: np.ndarray, 
                       v2: np.ndarray=0) -> float:
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
                      gamma: float=1) -> float:
    """
    Gaussian distance function
    """
    return np.exp(-gamma * euclidean_distance(v1, v2)**2)
    


def radial_basis_function_distance(v1: np.ndarray, 
                                   v2: np.ndarray, 
                                   gamma: float=1) -> float:
    r"""
    The Gaussian RBF kernel 
    $exp (\gama ||v1 - v2||^2)$.
    """
    
    return np.exp(-para_gamma * euclidean_distance(v1, v2))



def most_likely_estimator(concept: np.ndarray, 
                          bag: np.ndarray, 
                          sigma: float,
                          distance: str='euclidean') -> float:
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
    concept: (np.ndarray)
    bag: (np.ndarray)
    sigma: (float)
    """
    
    if distance == 'euclidean':
        similarity = np.exp(
            -1 * np.square(
                np.linalg.norm(bag - concept, axis=1)) / 
            (sigma ** 2)
            );
    elif distance == 'rbf':
        raise NotImplementedError("Other distance functions are not implemented");
    elif distance == 'rbf2':
        raise NotImplementedError("Other distance functions are not implemented");
    elif distance == 'gaussian':
        raise NotImplementedError("Other distance functions are not implemented");
    else:
        raise NotImplementedError("Other distance functions are not implemented");
    
    return max(similarity)


def embed_bag(concept_class: np.ndarray, 
              bag: np.ndarray, 
              sigma: float,
              distance: str='euclidean') -> np.ndarray:
    
    embedded_bag = np.zeros((concept_class.shape[0]))
    for n in range(0, embedded_bag.shape[0]):
        # Calculate similarity measure for each training instance
        embedded_bag[n] = most_likely_estimator(concept_class[n],
                                                bag,
                                                sigma,
                                                distance);
    
    return embedded_bag



def embed_all_bags(concept_class: np.ndarray,
                   bags: np.ndarray,
                   sigma: float,
                   distance: str='euclidean') -> np.ndarray:
    """Embed a set of bags onto a concept class (A concept class is a set of 
    instances)
    inputs
    -------
    outputs
    -------
    embedding: (np.ndarray) size (k,l) where k is the number of concepts in
    the concept class, and l is the number of bags being embedded. Each column 
    represents a bag, and the kth feature in the concept class realizes the kth 
    row of the matrix"""
    
    # Embed all bags using all training instances
    embedded_bags = np.zeros((concept_class.shape[0], bags.shape[0]))
    for i, bag in enumerate(bags):
        embedded_bags[:,i] = embed_bag(concept_class, bag, sigma, distance)
    
    return embedded_bags



