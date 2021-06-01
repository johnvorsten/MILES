# -*- coding: utf-8 -*-
"""
Created on Mon May 31 18:40:34 2021

@author: vorst
"""

# Python imports

# Third party imports
import sklearn as skl

# Local imports
from embedding import embed_all_bags
from embedding_test import generate_dummy_data

# Globals
N_POSITIVE_BAGS = 25
N_NEGATIVE_BAGS = 25
INSTANCE_SPACE = 4
N_BAGS = 9
gamma = 3 # Regularizer, Embedding
penalty = 'l1' # L1 loss penalization
loss = 'squared_hinge' # Loss function
C = 1.0 # SVM regularization, inversely proportional



#%% 

# Import
positive_bags, negative_bags = generate_dummy_data(N_BAGS, 
                                                   INSTANCE_SPACE,
                                                   N_POSITIVE_BAGS,
                                                   N_NEGATIVE_BAGS)

# Load
# Not required

# Transform
# Not required

# Define SVM
svmc = skl.svm.LinearSVC(loss=loss, penalty=penalty, C=C)
svm = skl.svm.NuSVC(loss=loss, penalty=penalty, C=C)


def main():
    pass

if __name__ == 'main':
    main()