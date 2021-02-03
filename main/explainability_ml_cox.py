# %%
# Explainability vs CPH
# In this script, we will compare the results of different ML methods
# for CPH for predicting breast cancer survival using SHAP, as described
# in the paper "Explainable Machine Learning Can Outperform Cox Regression 
# Predictions and Provide Insights in Breast Cancer Survival"
#
# Preliminaries
#
# Import packages

# %%
import os
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import random
from joblib import load, dump

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.metrics import make_scorer

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.svm import FastKernelSurvivalSVM

import xgboost

# Model explainability
import shap
shap.initjs()

import survival_analysis

PATH_SCRIPTS = os.path.join('..', 'scripts')
if PATH_SCRIPTS not in sys.path:
    sys.path.append(PATH_SCRIPTS)
import get_data
import helpers
import viz

# For reproducibility
SEED = 1601
random.seed(SEED) 


# %%
# Define paths

# %%
PATH_DATA = pathlib.Path(r'../data')
PATH_PARAMETERS = pathlib.Path(r'../parameters')
PATH_RESULTS = pathlib.Path(r'../results')

# Make sure directories exist
if not PATH_PARAMETERS.exists():
    PATH_PARAMETERS.mkdir(parents=True)
if not PATH_RESULTS.exists():
    PATH_RESULTS.mkdir(parents=True)


# %%
# Set some required/useful options/settings.


# %%
# %matplotlib inline
sns.set(style="whitegrid")
pd.set_option('display.max_columns', 50)

mpl.rcParams['font.sans-serif'] = "Calibri"
mpl.rcParams['font.family'] = "sans-serif"
sns.set(font_scale=1.75)
sns.set_style('ticks')
plt.rc('axes.spines', top=False, right=False)


# %%
# Import data

# %%
df = get_data.survival(PATH_DATA)


# %% 
# Get T/C.

# %%
df, (T, C) = survival_analysis.get_T_C(df)


# %%
# Define columns of interest
#
# Based on the feature selection stage, the most predictive features were
# (from most to least):
#
# - `age` - Age
# - `ratly` - Ratio between positive and removed lymph nodes
# - `rly` -Removed lymph nodes
# - `ptmm` - Tumor size [mm]
# - `pts` - Pathological tumor stage
#     - 1 - Stage 1
#     - 4 - Stage 2A
#     - 5 - Stage 2B
#     - 6 - Stage 3A
#     - 7 - Stage 3B
#     - 8 - Stage 3C
# - `grd` - Tumor grade
#     - 1 - Grade 1
#     - 2 - Grade 2
#     - 3 - Grade 3
#     - 4 - Grade 4
#
# Based on clinical expertise, the most predictive features should be
# (in no particular order):
#
# - `age` - Age
# - `ptmm` - Tumor size [mm]
# - `grd` - Tumor grade
#     - 1 - Grade 1
#     - 2 - Grade 2
#     - 3 - Grade 3
#     - 4 - Grade 4
# - `mor` - Tumor morphology
#     - 1 - Ductal
#     - 2 - Lobular
#     - 3 - Mixed
#     - 4 - Other
# - `ply` - Positive lymph hodes
# - `horm` - Hormone receptor status
#     - 1 - Triple negative
#     - 2 - Hormone receptor negative
#     - 3 - Hormone receptor positive
#     - 4 - Triple positive

# %%
cols_interest = ['age', 'ratly', 'rly', 'ptmm', 'pts', 'grd', 'mor', 'ply', 'horm']
df = df[cols_interest]


# %% 
# Perform classic CPH
# We will print the table of coefficients as well as generate a visual representation.

#%%

# Encode categorical values accordingly.
df_dummies = df.copy(deep=True)
df_dummies = helpers.decode(df_dummies)
df_dummies = pd.get_dummies(df_dummies, columns=['pts', 'grd', 'mor', 'horm'])

# Drop reference columns.
df_dummies.drop(columns=['pts_1', 'grd_1', 'mor_lobular', 'horm_3+'], inplace=True)

# Fix column ordering
columns = list(df_dummies.columns.values)
columns.append('horm_3-')
columns.remove('horm_3-')

# Visual representation.
fig, ax = survival_analysis.cph(df_dummies, T, C, latex_table=True, column_order=columns)
fig.savefig(PATH_RESULTS/'cph.pdf', dpi=300, bbox_inches='tight')


# %% 
# Model predictions of time to death
# We will compare their performance by computing the c-index.
# For this, we need to create a couple of things, including
# our own scorer.

#%%
c_index = {} # We will store c-index results here.
c_index_scorer = make_scorer(survival_analysis.c_index)


#%%
# We will compute the c-index using a 10-fold CV.

#%%
n_splits = 10
k_fold = KFold(n_splits=n_splits, random_state=SEED)

#%% 
# Convert data to the right format
# scikit-surv requires a specific format. Since we will use it
# quite a lot, it is convenient to do so properly at this point.

# %%
T = T*7 # weeks --> days

# DataFrame --> structured array
dataframe = {'censor': C, 'time':T} # series --> DataFrame
df_ss_tc = pd.DataFrame(dataframe)
s = df_ss_tc.dtypes
y_ss = np.array([tuple(x) for x in df_ss_tc.values], dtype=list(zip(s.index, s)))


#%%
# Conventional CPH

#%%
X_ss = df.copy(deep=True)

cph = CoxPHSurvivalAnalysis()
c_index['cph'] = cross_val_score(cph, X_ss, y_ss, cv=k_fold, scoring=c_index_scorer, verbose=1)


# %%
# Random Survival Forests (RSFs)

#%%
X_rsf = df.copy(deep=True)

rsf = RandomSurvivalForest(random_state=SEED, n_jobs=-1, verbose=True)
rsf.criterion = 'log_rank'

if not (PATH_PARAMETERS/'param_search_results_rsf.pkl').exists():
    helpers.save_best_params(rsf, X_rsf, y_ss, PATH_PARAMETERS, k_fold=k_fold, scorer=c_index_scorer, n_iter=25)

param_search_results_rsf = load(PATH_PARAMETERS/'param_search_results_rsf.pkl')
c_index['rsf'] = helpers.get_best_test_scores(param_search_results_rsf)


# %%
# Survival Support Vector Machine (SSVMs)

#%%
X_ssvm = df.copy(deep=True)

# Scale parameters (necessary for SSVM)
min_max_scaler = preprocessing.MinMaxScaler()
X_ssvm_scaled = min_max_scaler.fit_transform(X_ssvm)

ssvm = FastKernelSurvivalSVM(random_state=SEED, verbose=True)

if not (PATH_PARAMETERS/'param_search_results_ssvm.pkl').exists():
    helpers.save_best_params(ssvm, X_ssvm_scaled, y_ss, PATH_PARAMETERS, k_fold=k_fold, scorer=c_index_scorer, n_iter=25)

param_search_results_ssvm = load(PATH_PARAMETERS/'param_search_results_ssvm.pkl')
c_index['ssvm'] = helpers.get_best_test_scores(param_search_results_ssvm)

best_params_ssvm = param_search_results_ssvm.best_params_
ssvm.set_params(**best_params_ssvm)


# %% 
# Using XGBoost

# %%
X_xgb = df.copy(deep=True)
y_xgb = helpers.y_ss_to_xgb(y_ss)

#%%
xgb = xgboost.XGBRegressor(objective='survival:cox', nthread=-1)

if not (PATH_PARAMETERS/'param_search_results_xgb.pkl').exists():
    helpers.save_best_params(xgb, X_xgb, y_xgb, PATH_PARAMETERS, k_fold=k_fold, scorer=c_index_scorer)    

param_search_results_xgb = load(PATH_PARAMETERS/'param_search_results_xgb.pkl')
c_index['xgb'] = helpers.get_best_test_scores(param_search_results_xgb)

best_params_xgb = param_search_results_xgb.best_params_
xgb.set_params(**best_params_xgb)


# %% 
# Evaluation
# Visualize c-index results.

#%%
fig, ax = viz.plot_c_index(c_index)
fig.savefig(PATH_RESULTS/('c-index.pdf'), dpi=300, bbox_inches='tight')


# %%
# Explainability using SHAP.
# Due to the long computational time required for kernel SHAP, we will 
# only compare the reference case (CPH) and the best-performing ML model (XGB).
#
# First, we need to split our data and fit the models (using their
# best parameters).

#%%
X_ss_train, X_ss_test, y_ss_train, y_ss_test = train_test_split(X_ss, y_ss, test_size=1/n_splits, random_state=SEED)
cph.fit(X_ss_train, y_ss_train)

#%%
X_xgb_train, X_xgb_test, y_xgb_train, y_xgb_test = train_test_split(X_xgb, y_xgb, test_size=1/n_splits, random_state=SEED)
xgb.set_params(**best_params_xgb)
xgb.fit(X_xgb_train, y_xgb_train)


# %% 
# Then, we compute the SHAP values.
#
# In the case of CPH (i.e., when using SHAP's Kernel Explainer), 
# this can be VERY slow. Be careful! Therefore, we will compute it just once, save it,
# and load it from memory.

#%%
if not (PATH_RESULTS/'shap_chp.pkl').exists():

    cph_shap_explainer = shap.KernelExplainer(cph.predict, X_ss_test)
    cph_shap_values = cph_shap_explainer.shap_values(X_ss_test)    
    dump(cph_shap_values, PATH_RESULTS/'shap_chp.pkl')
    
cph_shap_values = load(PATH_RESULTS/'shap_chp.pkl')

#%%
fig, ax = viz.plot_shap(cph_shap_values, X_ss_test)
fig.savefig(PATH_RESULTS/('shap_cph.pdf'), dpi=300, bbox_inches='tight')

for col in X_ss_test.columns:
    fig, ax = viz.plot_shap_dependence(col, [cph_shap_values], X_ss_test)
    fig.savefig(PATH_RESULTS/f'shap_cph_{col}.pdf', dpi=300, bbox_inches='tight')
    
    
#%%
xgb_shap_explainer = shap.TreeExplainer(model=xgb)
xgb_shap_values = xgb_shap_explainer.shap_values(X_xgb_test)    

#%%
fig, ax = viz.plot_shap(xgb_shap_values, X_xgb_test)
fig.savefig(PATH_RESULTS/('shap_xgb.pdf'), dpi=300, bbox_inches='tight')

for col in X_xgb_test.columns:
    fig, ax = viz.plot_shap_dependence(col, [xgb_shap_values], X_xgb_test)
    fig.savefig(PATH_RESULTS/f'shap_xgb_{col}.pdf', dpi=300, bbox_inches='tight')
    
    
#%% 
# Furthermore, we can plot SHAP dependence values of both models
# for easier comparison.
    
#%%
for col in X_xgb_test.columns:
    fig, ax = viz.plot_shap_dependence(col, [cph_shap_values, xgb_shap_values], X_xgb_test)
    fig.savefig(PATH_RESULTS/f'shap_cph_xgb_{col}.pdf', dpi=300, bbox_inches='tight')
    
    
#%% 
# Interaction effects

#%%
xgb_shap_values_interaction = xgb_shap_explainer.shap_interaction_values(X_xgb_test)

#%%
fig, ax = viz.plot_shap_interactions(xgb_shap_values_interaction, X_xgb_test)
fig.savefig(PATH_RESULTS/('shap_xgb_interactions.pdf'), dpi=300, bbox_inches='tight')


#%%
for col in X_xgb_test.columns:
    fig, ax = viz.plot_shap_dependence_int(col, xgb_shap_values, X_xgb_test)
    fig.savefig(PATH_RESULTS/f'shap_xgb_{col}_int.pdf', dpi=300, bbox_inches='tight')
    