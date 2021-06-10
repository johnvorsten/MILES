# -*- coding: utf-8 -*-
"""
Created on Mon May 31 18:40:34 2021

@author: vorst
"""

# Python imports

# Third party imports
import sklearn as skl
from sklearn import model_selection
from sklearn import svm
import numpy as np

# Local imports
from embedding import embed_all_bags, generate_dummy_data

# Globals
N_POSITIVE_BAGS = 100
N_NEGATIVE_BAGS = 25
INSTANCE_SPACE = 4
BAG_SIZE = 9
SIGMA_EMBEDDING = 3 # Regularizer, Embedding
GAMMA_SVC = 'scale'
PENALTY = 'l1' # L1 loss penalization
LOSS = 'squared_hinge' # Loss function
C = 1.0 # SVM regularization, inversely proportional

#%% 

# Import
positive_bags, negative_bags = generate_dummy_data(BAG_SIZE, 
                                                   N_POSITIVE_BAGS,
                                                   N_NEGATIVE_BAGS)
concept_class_positive, concept_class_negative = \
    generate_dummy_data(BAG_SIZE, 
                        N_POSITIVE_BAGS,
                        N_NEGATIVE_BAGS)
    
bags = np.concatenate((positive_bags, negative_bags), axis=0)
labels = np.concatenate((np.ones(N_POSITIVE_BAGS), np.zeros(N_NEGATIVE_BAGS)), axis=0)
concept_class = np.concatenate((concept_class_positive, concept_class_negative), axis=0)
concept_labels = np.concatenate((np.ones(N_POSITIVE_BAGS), np.zeros(N_NEGATIVE_BAGS)), axis=0)


# Load / split data
train, test, train_label, test_label = \
    skl.model_selection.train_test_split(bags, 
                                         labels,
                                         test_size=0.15)

# Transform & Encode
embedded_bags = embed_all_bags(concept_class=concept_class, 
                               bags=bags,
                               sigma=SIGMA_EMBEDDING,
                               distance='euclidean')

# Define SVM
svmc_l1 = skl.svm.LinearSVC(loss=LOSS, penalty=PENALTY, C=C, dual=False, max_iter=5000)

# SVC Using LibSVM uses the squared l2 loss
svmc = skl.svm.SVC(kernel='rbf', gamma=GAMMA_SVC, C=C)

# Define grid search parameters
params_l1svc = {'C':[0.5,1,2],
                }
params_svc = {'C':[0.5,1,2],
              'kernel':['rbf', 'poly']}

# Grid search
svmc_l1_gs = skl.model_selection.GridSearchCV(estimator=svmc_l1,
                                 param_grid=params_l1svc,
                                 scoring=['accuracy','precision','recall'],
                                 refit=False,
                                 n_jobs=6,
                                 cv=None, # Default 5-fold validation
                                 )


svmc_gs = skl.model_selection.GridSearchCV(estimator=svmc,
                                 param_grid=params_svc,
                                 scoring=['accuracy','precision','recall'],
                                 n_jobs=6,
                                 refit=False,
                                 cv=None, # Default 5-fold validation
                                 )


if __name__ == 'main':
    # Estimators expect (instance,features). embedded bags are encoded where 
    # features are along axis=1
    svmc_l1_gs.fit(np.transpose(embedded_bags), labels)
    svmc_gs.fit(np.transpose(embedded_bags), labels)
    
    # Print results
    print("L1 SVM Results: ", svmc_l1_gs.cv_results_, "\n\n")
    print("rbf, polynomial SVM Results: ", svmc_gs.cv_results_, "\n\n")






