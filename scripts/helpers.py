# -*- coding: utf-8 -*-
"""
helpers.py
Auxiliary functions.

Created on Thu Mar 19 12:34:34 2020
"""
# Import packages.
import pathlib
import numpy as np
from joblib import dump

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
#from sklearn.utils.fixes import loguniform
from sksurv.ensemble import RandomSurvivalForest
from sksurv.svm import FastKernelSurvivalSVM

import xgboost


#%%
def get_model_type(model):
    """
    Get a string representing the name of a model.

    Parameters
    ----------
    model: model object
        Model object.

    Returns
    -------
    model_type: string
        (Abbreviated) string corresponding to the model type.
        Current supported models are:
            'rsf'   sksurv.ensemble.RandomSurvivalForest
            'ssvm'  sksurv.svm.FastKernelSurvivalSVM
            'xgb'   xgboost.XGBRegressor or xgboost.XGBClassifier
    
    """  
    if isinstance(model, RandomSurvivalForest):
        model_type = 'rsf'
    elif isinstance(model, FastKernelSurvivalSVM):
        model_type = 'ssvm'
    elif (isinstance(model, xgboost.XGBRegressor)) or (isinstance(model, xgboost.XGBClassifier)):
        model_type = 'xgb'
    else:
        model_type = None
        raise Exception("Invalid model type. For supported models, see docstring.")
        
    return model_type
 
    
#%%        
def save_best_params(model, X, y, path_params_file, scorer=None, k_fold=None, n_iter=25, verbose=True):
    """
    Get an optimized set of parameters for a specific model.
    They are calculated using randomized search with cross validation and then
    saved to a .pkl file.

    Parameters
    ----------
    model:
        Original model whose parameters will be optimized.
        
    X: pandas DataFrame
        Model features
        
    y: list
        Model outputs
        
    path_params_file: string or Pathlib path
        Path where the parameters file will be saved (or loaded)
        
    scorer: string or scikit's scorer object
        Scorer used to optimize during the random search.
        If None, it will use 'accuracy'
        
    k_fold: scikit's KFold object
        k_fold object to be used.
        If None, one will be created with 10 folds.
        
    n_iter: integer
        Number of searches that will be performed.
        Notice that the total number of iterations will be given by
        the number of folds (10 by default) times n_iter.
        
    verbose: boolean (optional)
        Define if verbose output is desired (True, default) or not (False)

    Returns
    -------
    None
    
    Outputs
    -------
    Writes file "param_search_results_MODEL.pkl".
    """    


    #%% 
    # Validate inputs
    if isinstance(path_params_file, str):
        path_params_file = pathlib.Path(path_params_file)
        
    # Check for model type.
    model_type = get_model_type(model)
    if verbose:
        print(f"Model type is {model_type}")
        
    
    #%% 
    # Define parameters to explore.
    if verbose:
        print("Definining parameters... ", end="", flush=True)
        
    if model_type == 'rsf':
        n_estimators = [25, 50, 75, 100, 250, 500, 750, 1000, 1500] # Number of trees
        max_depths = [1, 2, 3, 4, 5, 10, 15, 25, 50, 100, 250, 500] # Maximum depth of the tree
        min_samples_splits = [5, 10, 15, 20, 25, 50] # Minimum number of samples required to split an internal node
        min_samples_leafs = [1, 2, 3, 4, 5, 10, 25] # Minimum number of samples required to be at a leaf node

        param_grid = dict(n_estimators=n_estimators,
                          max_depth=max_depths,
                          min_samples_split=min_samples_splits,
                          min_samples_leaf=min_samples_leafs)
        
        
    elif model_type == 'ssvm':
        kernels = ['linear', 'rbf', 'sigmoid']
        alphas = np.logspace(-5, 1, num=1000)
        gammas = np.logspace(-3, 0, num=1000)
        
        param_grid = dict(kernel=kernels,
                          alpha=alphas,
                          gamma=gammas)
        
        
    elif model_type == 'xgb':
        boosters = ['gbtree']
        max_depths = [1, 2, 3, 4, 5, 10, 15, 25, 50, 100, 250, 500]
        n_estimators = [25, 50, 75, 100, 250, 500, 750, 1000, 1500]
        learning_rates = np.logspace(-2, 0, num=1000)
        subsamples = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        
        param_grid = dict(booster=boosters,
                          max_depth=max_depths, 
                          n_estimator=n_estimators,
                          learning_rate=learning_rates,
                          subsample=subsamples)
        
    else:
        raise Exception("Invalid model type. Currently, only rsf, ssvm, and xgb are supported")
    
    if verbose:
        print(param_grid)
        print("DONE!")
    
    #%%
    if verbose:
        print("Creating parameter search objects... ", end="", flush=True)
        
    # If objects were not given as parameters, create them.
    if k_fold == None:
        k_fold = KFold(n_splits=10)
    if scorer == None:
        scorer = 'accuracy'

    grid_search = RandomizedSearchCV(model, param_grid, scoring=scorer, cv=k_fold, n_iter=n_iter, error_score=np.nan, n_jobs=-1, verbose=1)    
    if verbose:
        print("DONE!")
               
    #%%
    if verbose:
        print("Performing parameter search... ", end="", flush=True)
    grid_result = grid_search.fit(X, y)
    if verbose:
        print("DONE!")
        
    #%%
    if verbose:
        print("Saving parameter search results...", end="", flush=True)
    dump(grid_result, path_params_file/f'param_search_results_{model_type}.pkl')
    if verbose:
        print("DONE!")

    return None


#%%        
def get_best_test_scores(search_cv_results):
    """
    Get the raw test scores of the best parameter set of a
    GridSearchCV or a RandomSearchCV object.
    
    Note that although we can get the mean and SD of these directly
    (search_cv_results.cv_results_.mean_test_score or
    search_cv_results.cv_results_.std_test_score), we need to get the
    actual values for plotting and statistical testing.

    Parameters
    ----------
    search_cv_results: scikit GridSearchCV or RandomizedSearchCV object
        Parameter search results.

    Returns
    -------
    best_test_scores: numpy array
        Array with the test scores of the best parameter set.
    
    References
    -------
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html
    """
    
    if (not isinstance(search_cv_results, GridSearchCV)) and (not isinstance(search_cv_results, RandomizedSearchCV)):
        raise Exception("search_cv_results input is not an instance of GridSearchCV or RandomizedSearchCV.")
        
    # Get best run.
    best_run_index = search_cv_results.best_index_
        
    # Get number of splits used in CV.
    n = search_cv_results.n_splits_
    
    # Get best test scores.
    best_test_scores = []
    for ii in range(0, n):        
        best_test_scores.append(search_cv_results.cv_results_[f'split{ii}_test_score'][best_run_index])

    return np.asarray(best_test_scores)



#%%
def decode(df):
    """
    Decode variables to explicit values.
    
    Parameters
    ----------
    df: pandas DataFrame
        Data frame to decode.

    Returns
    -------
    df_decoded: pandas DataFrame
        Decoded DataFrame.
    """

    df_decoded = df.copy(deep=True)

    if 'pts' in df.columns:
        d = {1:'1',
             4:'2A',
             5:'2B',
             6:'3A',
             7:'3B',
             8:'3C'
             }
        df_decoded['pts'] = df['pts'].map(d)
        
    if 'mor' in df.columns:
        d = {1:'Ductal',
             2:'Lobular',
             3:'Mixed',
             4:'Other'
             }
        df_decoded['mor'] = df['mor'].map(d)
        
    if 'horm' in df.columns:
        d = {1:'3-',
             2:'HR-',
             3:'HR+',
             4:'3+'
             }
        df_decoded['horm'] = df['horm'].map(d)
        
    return df_decoded


#%%
def y_ss_to_xgb(y_ss):
    """
    Convert y from scikit-surv to xgb format.
    
    Parameters
    ----------
    y_ss: structured array
        y array used by scikit surv:
        a structured array containing the binary event indicator as first 
        field, and time of event or time of censoring as second field. 

    Returns
    -------
    y_xgb: list
        Negative numbers correspond to censored times.
        Positive numbers correspond to uncensored times.
    """

    y_xgb = [x[1] if x[0] else -x[1] for x in y_ss]
    
    return y_xgb
