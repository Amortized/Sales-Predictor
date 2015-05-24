"""

__author__ = 'amortized'
"""

import numpy  as np;
from sklearn.preprocessing import Imputer;
from sklearn.grid_search import ParameterGrid;
from multiprocessing import Pool;
import copy;
import random;
import sys;
import warnings;
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix;
import matplotlib.pyplot as plt;
import numpy as np
import scipy.sparse
import pickle
import xgboost as xgb
import copy
from sklearn.preprocessing import OneHotEncoder




def do(train_X, train_Y, test_X, test_ids):
    categorical_features_1 = [i for i in range(19,50)];
    categorical_features_2 = [i for i in range(51,60)];

    categorical_features = [];
    categorical_features.extend([16,18]);
    categorical_features.extend(categorical_features_1);
    categorical_features.extend(categorical_features_2);


    imp     = Imputer(missing_values='NaN', strategy='median', axis=0);
    enc     = OneHotEncoder(n_values='auto', categorical_features=np.array(categorical_features));

    merged  = train_X + test_X;

    imp.fit(train_X);
    enc.fit(merged);

    #Impute the  data
    train_X = imp.transform(train_X);
    test_X  = imp.transform(test_X);

    #Encode the data
    train_X = enc.tranform(train_X);
    test_X  = enc.tranform(test_X);

    print(train_X)
    print(test_X)






