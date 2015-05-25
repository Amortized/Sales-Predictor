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
from random import randint
from random import shuffle
import math

def generateParams():
    # Set the parameters by cross-validation
    paramaters_grid    = {'eta': [0.07], 'min_child_weight' : [6],  'colsample_bytree' : [0.95], 'subsample' : [0.95], 'gamma' : [0], 'max_depth' : [8]};

    paramaters_search  = list(ParameterGrid(paramaters_grid));

    parameters_to_try  = [];
    for ps in paramaters_search:
        params           = {'eval_metric' : 'rmse', 'objective' : 'reg:linear', 'nthread' : 8};
        for param in ps.keys():
            params[str(param)] = ps[param];
        parameters_to_try.append(copy.copy(params));

    return parameters_to_try;     

def evaluate(Y, Y_hat):
    Y_hat = [0 if yh < 0 else yh for yh in Y_hat]
    Y     = [0 if y < 0 else y for y in Y]
    diff = [ math.pow( (math.log(Y_hat[i]+1) - math.log(Y[i]+1)), 2)  for i in range(0, len(Y)) ];
    return math.sqrt( sum(diff) / float(len(diff)));



def build(features, label):
    X_train, X_validation, Y_train, Y_validation = train_test_split(features, label, test_size=0.10, random_state=100);

    #Load Data
    dtrain      = xgb.DMatrix( X_train, label=Y_train);
    dvalidation = xgb.DMatrix( X_validation, label=Y_validation);

    parameters_to_try = generateParams();

    best_model     = None;
    best_score     = sys.float_info.max;
    best_iteration = 0;
    best_params    = None;


    for i in range(0, len(parameters_to_try)):
        param     = parameters_to_try[i]
        #Train a Model
        evallist  = [(dtrain,'train'), (dvalidation,'eval')]
        num_round = 1000
        bst       = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=10)

        #Get a score
        Y_hat = bst.predict( dvalidation ,ntree_limit=bst.best_iteration)
        current_score = evaluate(Y_validation, Y_hat);

        if current_score < best_score:
            best_score     = current_score;
            best_iteration = bst.best_iteration;
            best_model     = bst
            best_params    = param;


    print("Validation Score " + str(best_score));
    print("Best Params "      + str(best_params));
    print("Best Iteration "   + str(best_iteration));
    best_model.save_model('./data/best_model.model');

    return best_iteration;

def predict_and_write(model1_path, test_X, test_ids, model1_best_iteration):
    #Load model
    model1 = xgb.Booster({'nthread':8})
    model1.load_model(model1_path)

    test_X = xgb.DMatrix(test_X);
    f = open("./data/submission.csv", "w");
    f.write("id,units\n");

    
    Y_hat = model1.predict( test_X ,ntree_limit=model1_best_iteration);
    Y_hat = [0 if yh < 0 else yh for yh in Y_hat]

    for i in range(0, len(Y_hat)):
       f.write(str(test_ids[i]) + "," + str(Y_hat[i]) + "\n");

    f.close();

    

def do(train_X, train_Y, test_X, test_ids):
    categorical_features_1 = [i for i in range(19,50)];
    categorical_features_2 = [i for i in range(51,60)];

    categorical_features = [];
    categorical_features.extend([16,18]);
    categorical_features.extend(categorical_features_1);
    categorical_features.extend(categorical_features_2);


    imp     = Imputer(missing_values='NaN', strategy='median', axis=0);
    enc     = OneHotEncoder(n_values='auto', categorical_features=np.array(categorical_features), sparse=False);

    
    #Impute the  data
    imp.fit(train_X);

    train_X = imp.transform(train_X);
    test_X  = imp.transform(test_X);	

    merged  = np.concatenate((train_X, test_X), axis=0);

    enc.fit(merged);

    #Encode the data
    train_X = enc.transform(train_X);
    print("Imputation and encoding completed.");


    best_iteration = build(train_X, train_Y);

    del train_X, train_Y;
    test_X = enc.transform(test_X);

    predict_and_write('./data/best_model.model', test_X, test_ids, best_iteration)






