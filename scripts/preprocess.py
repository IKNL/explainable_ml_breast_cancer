# -*- coding: utf-8 -*-
"""
preprocess.py
Functions for preprocessing data.

Created on Tu Mar 3 10:41:26 2020
"""

# %%
# Import packages.
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# Set some required/useful options and settings
import operator, math, datetime
from itertools import combinations

# %matplotlib inline
sns.set(style="white")
pd.set_option('display.max_columns', 50)

import warnings
warnings.filterwarnings('ignore')

from knn_impute import knn_impute

# %%  
# Get the date of today formatted like "16Jan" to be used for saving/loading of constructed models
date_notation = datetime.date.today().strftime("%d%b")

# %%  
# Import the raw dataset with missing values from the Data Prep step
df = pd.read_csv("../data/df_raw.csv")

# %%  
df.info()

# %%  
# Inspecting missing values

num_rows = df.shape[0]
num_columns = df.shape[1]
all_values =  num_rows * num_columns 
missing_values = df.isna().sum().sum()

print("The final dataframe consists of {} number of unique patients with {} features containing information about \
them".format(num_rows, num_columns))
print("A full dataframe would thus contain {} values. Currently, the dataframe contains {} missing values, \
which is {:.2f}% of all possible values.".format(all_values, missing_values, (missing_values/all_values)*100))

# %%  
# Retrieve a list of which columns contain missing values
cols_with_na = df.columns[df.isna().any()].tolist()
cols_with_na

# %%  
# To immediately get an insight into the missing values for different time windows, sort the dataframe 
# based on year of incidence
df_sorted = df.sort_values(by=["inc"])

# %%
ax = msno.matrix(df = df_sorted, figsize = (25, 15), color=(0.2, 0.2, 0.2), sparkline=False, fontsize=24)

start, end = ax.get_ylim()
stepsize = num_rows / len(df_sorted.inc.unique())
ax.yaxis.set_ticks(np.arange(end, start, stepsize))
ax.yaxis.set_ticklabels(list(df_sorted.inc.unique()))

plt.show()

# %%  
# The second plot will be a heatmap of the correlations between missing values in the dataset. 
msno.heatmap(df_sorted)
plt.savefig("../visualizations/msno_heatmap_{}.png".format(date_notation))

# %%  
# Third, a dendogram is plotted to visualize the correlation of variable completion in more detail.
msno.dendrogram(df_sorted)
plt.savefig("../visualizations/msno_dendrogram_{}.png".format(date_notation))

# %%  
df_missing = df.isna()

# %%  
df_missing_sum = df_missing.sum()

missing_values = df_missing_sum[df_missing_sum != 0]
missing_values_perc = missing_values / len(df)

# %%
missing_columns = list(missing_values.keys())
missing_columns_nums = missing_values.values

ind = np.arange(1, len(missing_columns) + 1)

plt.figure(figsize=(20,6))
ax = plt.subplot(111)
ax.bar(ind, missing_columns_nums, tick_label = missing_columns, color="orange")

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.xticks(rotation=45, fontsize=14)
plt.xlabel("Feature", fontsize=16)
plt.ylabel("Number of missing values (percentage)", fontsize=16)
plt.title("")

for i, v in enumerate(missing_columns_nums):
    
    if v < 1000:
        y = v + 3500
    elif v < 10000:
        y = v + 6000
    elif v < 100000:
        y = v + 7500
    else:
        y = v - 50000
        
    perc = round((v/len(df)*100), 1)
    text = str(v) + " (" + str(perc) + "%)"
    
    plt.text(ind[i] - 0.125, y, text, rotation = 90, color="black", size=15)

plt.savefig("../visualizations/full_df_missing_values_{}.png".format(date_notation))
plt.show()

# %%
# DL imputation

# Compute the Pearson correlation matrix
pearson_corr = df.corr(method="pearson")

# Generate a mask for the upper triangle
mask = np.zeros_like(pearson_corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(24, 24))

# Draw the heatmap with the mask, colormap and show the correlation percentages
hm = sns.heatmap(pearson_corr, annot=True, annot_kws={"size": 10}, mask=mask, cmap="coolwarm", square=True, 
                 linewidths=.5, fmt= '.2f', ax=ax, cbar_kws={"shrink": .5})

hm.set_yticklabels(hm.get_yticklabels(), rotation = 0)
hm.set_xticklabels(hm.get_xticklabels(), rotation = 90)

plt.show()

# %%
# Compute the Spearman correlation matrix
spearman_corr = df.corr(method="spearman")

# Generate a mask for the upper triangle
mask = np.zeros_like(spearman_corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(24, 24))

# Draw the heatmap with the mask, colormap and show the correlation percentages
hm = sns.heatmap(spearman_corr, annot=True, annot_kws={"size": 10}, mask=mask, cmap="coolwarm", square=True, 
                 linewidths=.5, fmt= '.2f', ax=ax, cbar_kws={"shrink": .5})

hm.set_yticklabels(hm.get_yticklabels(), rotation = 0)
hm.set_xticklabels(hm.get_xticklabels(), rotation = 90)

plt.show()

# %%
# Compute the Kendall correlation matrix
kendall_corr = df.corr(method="kendall")

# Generate a mask for the upper triangle
mask = np.zeros_like(kendall_corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(24, 24))

# Draw the heatmap with the mask, colormap and show the correlation percentages
hm = sns.heatmap(kendall_corr, annot=True, annot_kws={"size": 10}, mask=mask, cmap="coolwarm", square=True, 
                 linewidths=.5, fmt= '.2f', ax=ax, cbar_kws={"shrink": .5})

hm.set_yticklabels(hm.get_yticklabels(), rotation = 0)
hm.set_xticklabels(hm.get_xticklabels(), rotation = 90)

plt.show()

# %%  
from random import choice
import datawig
from datawig.utils import random_split
from sklearn.model_selection import train_test_split
from datawig.column_encoders import NumericalEncoder, CategoricalEncoder
from datawig.mxnet_input_symbols import NumericalFeaturizer, LSTMFeaturizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

# %%  
# Since the data contains many categorical variables, we temporarily map them back to a categorical string variable 
# to ensure they are treated as categorica variables by the Imputer. 
categorical_dict = {0: "Z", 1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F", 7: "G", 8: "H", 9: "I", 10: "J", 11: "K", 
                    12: "L", 13: "M", 14: "N", 15: "O", 16: "P", 17: "Q", 18: "R", 19: "S", 20: "T", 21: "U", 22: "V", 
                    23: "W", 24: "X", 25: "Y"}

inv_categorical_dict = {"Z": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9, "J": 10, "K": 11, 
                        "L": 12, "M": 13, "N": 14, "O": 15, "P": 16, "Q": 17, "R": 18, "S": 19, "T": 20, "U": 21, "V": 22, 
                        "W": 23, "X": 24, "Y": 25}

# %%  
# Define a list of categorical features to loop through later and format them as explicitly categorical
# The feature "morf" is not included as this one needs a special mapping and will be treated separately
categorical_features = ["vit", "loc", "lat", "mor", "beh", "grd", "diag", "ct", "cn", "cm", "pt", "pn", "pm", "cts", 
                        "pts", "fh", "trh", "ftrh", "surg", "neoct", "neott", "neoht", "ER", "HER2", "PR", "men"]

# %%  
for feature in categorical_features:
    df[feature] = df[feature].map(categorical_dict)

# %%  
df[categorical_features].sample(7)

# %%  
cat_indices = ["Accuracy", "Average Precision", "Weighted Precision", "Average Recall", "Weighted Recall", 
               "Average F1 Score", "Weighted F1 Score"]
cont_indices =  ["Mean Squared Error", "Mean Absolute Error", "Explained Variance Score", "R2 Score"]

# %%
# Retrieve a list of which columns contain missing values
cols_with_na = df.columns[df.isna().any()].tolist()
cols_with_na

# %%
# Menopausal status (men) training and testing

print("PEARSON CORRELATIONS")
for key, value in pearson_corr.men[(pearson_corr.men >= 0.3) | (pearson_corr.men <= -0.3)].iteritems():
    if key != "men":
        print("Pearson correlation coefficient with {} is: {:.2f}.".format(key, value))

print("")
print("SPEARMAN CORRELATIONS")

for key, value in spearman_corr.men[(spearman_corr.men >= 0.3) | (spearman_corr.men <= -0.3)].iteritems():
    if key != "men":
        print("Spearman correlation coefficient with {} is: {:.2f}.".format(key, value))

print("")
print("KENDALL CORRELATIONS")

for key, value in kendall_corr.men[(kendall_corr.men >= 0.3) | (kendall_corr.men <= -0.3)].iteritems():
    if key != "men":
        print("Kendall correlation coefficient with {} is: {:.2f}.".format(key, value))

# %%  
fl = [["age"]]

men_scores = {}

for f in fl:
    
    for i in range(1, 4):
        df_train, df_test = train_test_split(df, test_size = 0.1)
        df_train, df_validation = train_test_split(df_train, test_size = 0.11113)
        
        #Initialize a SimpleImputer model
        imputer = datawig.SimpleImputer(
            input_columns=f,
            output_column="men",
            output_path ="../imputers/imputer_model"
        )

        #Fit an imputer model on the train data
        imputer.fit(train_df = df_train, test_df = df_validation, num_epochs = 3)

        #Impute missing values and return original dataframe with predictions
        imputed = imputer.predict(df_test)

        not_empty_df_test = imputed.loc[imputed.men.notnull()]

        not_empty_df_test.men = not_empty_df_test.men.map(inv_categorical_dict)
        not_empty_df_test.men_imputed = not_empty_df_test.men_imputed.map(inv_categorical_dict)

        acc = accuracy_score(not_empty_df_test.men, not_empty_df_test.men_imputed)
        avg_prec = precision_score(not_empty_df_test.men, not_empty_df_test.men_imputed, average="macro")
        weight_prec = precision_score(not_empty_df_test.men, not_empty_df_test.men_imputed, average="weighted")
        avg_rec = recall_score(not_empty_df_test.men, not_empty_df_test.men_imputed, average="macro")
        weight_rec = recall_score(not_empty_df_test.men, not_empty_df_test.men_imputed, average="weighted")
        avg_f1 = f1_score(not_empty_df_test.men, not_empty_df_test.men_imputed, average="macro")
        weight_f1 = f1_score(not_empty_df_test.men, not_empty_df_test.men_imputed, average="weighted")

        metrics = [acc, avg_prec, weight_prec, avg_rec, weight_rec, avg_f1, weight_f1]
        key = '_'.join(f) + "_cv_" + str(i)

        men_scores[key] = metrics

# %%  
men_scores_matrix = pd.DataFrame(men_scores)
men_scores_matrix.index = cat_indices
men_scores_matrix = men_scores_matrix.T
men_scores_matrix

# %%  
men_scores_matrix.to_excel("../stats/imputation/men_scores_matrix_{}.xlsx".format(date_notation))

# %%  
men_imputer_columns = ["age"]

# %%
# Tumour size in millimeters (ptmm) imputation training and testing

print("PEARSON CORRELATIONS")
for key, value in pearson_corr.ptmm[(pearson_corr.ptmm >= 0.3) | (pearson_corr.ptmm <= -0.3)].iteritems():
    if key != "ptmm":
        print("Pearson correlation coefficient with {} is: {:.2f}.".format(key, value))

print("")
print("SPEARMAN CORRELATIONS")

for key, value in spearman_corr.ptmm[(spearman_corr.ptmm >= 0.3) | (spearman_corr.ptmm <= -0.3)].iteritems():
    if key != "ptmm":
        print("Spearman correlation coefficient with {} is: {:.2f}.".format(key, value))

print("")
print("KENDALL CORRELATIONS")

for key, value in kendall_corr.ptmm[(kendall_corr.ptmm >= 0.3) | (kendall_corr.ptmm <= -0.3)].iteritems():
    if key != "ptmm":
        print("Kendall correlation coefficient with {} is: {:.2f}.".format(key, value))

# %%  
fl = [["pts"], ["pt"], ["pts", "pt"]]

ptmm_scores = {}

for f in fl:
    
    for i in range(1, 4):
        df_train, df_test = train_test_split(df, test_size = 0.1)
        df_train, df_validation = train_test_split(df_train, test_size = 0.11113)
        
        #Initialize a SimpleImputer model
        imputer = datawig.SimpleImputer(
            input_columns=f,
            output_column="ptmm",
            output_path ="../imputers/imputer_model"
        )

        #Fit an imputer model on the train data
        imputer.fit(train_df = df_train, test_df = df_validation, num_epochs = 3)

        #Impute missing values and return original dataframe with predictions
        imputed = imputer.predict(df_test)

        not_empty_df_test = imputed.loc[imputed.ptmm.notnull()]

        mse = mean_squared_error(not_empty_df_test.ptmm, not_empty_df_test.ptmm_imputed)
        mae = mean_absolute_error(not_empty_df_test.ptmm, not_empty_df_test.ptmm_imputed)
        evs = explained_variance_score(not_empty_df_test.ptmm, not_empty_df_test.ptmm_imputed)
        r2 = r2_score(not_empty_df_test.ptmm, not_empty_df_test.ptmm_imputed)

        metrics = [mse, mae, evs, r2]
        key = '_'.join(f)  + "_cv_" + str(i)

        ptmm_scores[key] = metrics

# %%  
ptmm_scores_matrix = pd.DataFrame(ptmm_scores)
ptmm_scores_matrix.index = cont_indices
ptmm_scores_matrix = ptmm_scores_matrix.T
ptmm_scores_matrix

# %%  
ptmm_scores_matrix.to_excel("../stats/imputation/ptmm_scores_matrix_{}.xlsx".format(date_notation))

# %%  
ptmm_imputer_columns = ["pt"]

# %%
# ER and PR imputation training and testing

print("PEARSON CORRELATIONS")
for key, value in pearson_corr.ER[(pearson_corr.ER >= 0.3) | (pearson_corr.ER <= -0.3)].iteritems():
    if key != "ER":
        print("Pearson correlation coefficient with {} is: {:.2f}.".format(key, value))

print("")
print("SPEARMAN CORRELATIONS")

for key, value in spearman_corr.ER[(spearman_corr.ER >= 0.3) | (spearman_corr.ER <= -0.3)].iteritems():
    if key != "ER":
        print("Spearman correlation coefficient with {} is: {:.2f}.".format(key, value))

print("")
print("KENDALL CORRELATIONS")

for key, value in kendall_corr.ER[(kendall_corr.ER >= 0.3) | (kendall_corr.ER <= -0.3)].iteritems():
    if key != "ER":
        print("Kendall correlation coefficient with {} is: {:.2f}.".format(key, value))

# %%  
fl = [["PR"], ["grd"], ["PR", "grd"]]

ER_scores = {}

for f in fl:
    
    for i in range(1, 4):
        df_train, df_test = train_test_split(df, test_size = 0.1)
        df_train, df_validation = train_test_split(df_train, test_size = 0.11113)
        
        #Initialize a SimpleImputer model
        imputer = datawig.SimpleImputer(
            input_columns=f,
            output_column="ER",
            output_path ="../imputers/imputer_model"
        )

        #Fit an imputer model on the train data
        imputer.fit(train_df = df_train, test_df = df_validation, num_epochs = 3)

        #Impute missing values and return original dataframe with predictions
        imputed = imputer.predict(df_test)

        not_empty_df_test = imputed.loc[imputed.ER.notnull()]

        not_empty_df_test.ER = not_empty_df_test.ER.map(inv_categorical_dict)
        not_empty_df_test.ER_imputed = not_empty_df_test.ER_imputed.map(inv_categorical_dict)

        acc = accuracy_score(not_empty_df_test.ER, not_empty_df_test.ER_imputed)
        avg_prec = precision_score(not_empty_df_test.ER, not_empty_df_test.ER_imputed, average="macro")
        weight_prec = precision_score(not_empty_df_test.ER, not_empty_df_test.ER_imputed, average="weighted")
        avg_rec = recall_score(not_empty_df_test.ER, not_empty_df_test.ER_imputed, average="macro")
        weight_rec = recall_score(not_empty_df_test.ER, not_empty_df_test.ER_imputed, average="weighted")
        avg_f1 = f1_score(not_empty_df_test.ER, not_empty_df_test.ER_imputed, average="macro")
        weight_f1 = f1_score(not_empty_df_test.ER, not_empty_df_test.ER_imputed, average="weighted")

        metrics = [acc, avg_prec, weight_prec, avg_rec, weight_rec, avg_f1, weight_f1]
        key = '_'.join(f) + "_cv_" + str(i)

        ER_scores[key] = metrics

# %%  
ER_scores_matrix = pd.DataFrame(ER_scores)
ER_scores_matrix.index = cat_indices
ER_scores_matrix = ER_scores_matrix.T
ER_scores_matrix

# %%  
ER_scores_matrix.to_excel("../stats/imputation/ER_scores_matrix_{}.xlsx".format(date_notation))

# %%  
ER_imputer_columns = ["PR", "grd"]

# %%  
fl = [["ER"], ["grd"], ["ER", "grd"]]

PR_scores = {}

for f in fl:
    
    for i in range(1, 4):
        df_train, df_test = train_test_split(df, test_size = 0.1)
        df_train, df_validation = train_test_split(df_train, test_size = 0.11113)
        
        #Initialize a SimpleImputer model
        imputer = datawig.SimpleImputer(
            input_columns=f,
            output_column="PR",
            output_path ="../imputers/imputer_model"
        )

        #Fit an imputer model on the train data
        imputer.fit(train_df = df_train, test_df = df_validation, num_epochs = 3)

        #Impute missing values and return original dataframe with predictions
        imputed = imputer.predict(df_test)

        not_empty_df_test = imputed.loc[imputed.PR.notnull()]

        not_empty_df_test.PR = not_empty_df_test.PR.map(inv_categorical_dict)
        not_empty_df_test.PR_imputed = not_empty_df_test.PR_imputed.map(inv_categorical_dict)

        acc = accuracy_score(not_empty_df_test.PR, not_empty_df_test.PR_imputed)
        avg_prec = precision_score(not_empty_df_test.PR, not_empty_df_test.PR_imputed, average="macro")
        weight_prec = precision_score(not_empty_df_test.PR, not_empty_df_test.PR_imputed, average="weighted")
        avg_rec = recall_score(not_empty_df_test.PR, not_empty_df_test.PR_imputed, average="macro")
        weight_rec = recall_score(not_empty_df_test.PR, not_empty_df_test.PR_imputed, average="weighted")
        avg_f1 = f1_score(not_empty_df_test.PR, not_empty_df_test.PR_imputed, average="macro")
        weight_f1 = f1_score(not_empty_df_test.PR, not_empty_df_test.PR_imputed, average="weighted")

        metrics = [acc, avg_prec, weight_prec, avg_rec, weight_rec, avg_f1, weight_f1]
        key = '_'.join(f) + "_cv_" + str(i)

        PR_scores[key] = metrics

# %%  
PR_scores_matrix = pd.DataFrame(PR_scores)
PR_scores_matrix.index = cat_indices
PR_scores_matrix = PR_scores_matrix.T
PR_scores_matrix

# %%  
PR_scores_matrix.to_excel("../stats/imputation/PR_scores_matrix_{}.xlsx".format(date_notation))

# %%  
PR_imputer_columns = ["ER", "grd"]

# %%
# Surgery (surg) imputation training and testing

print("PEARSON CORRELATIONS")
for key, value in pearson_corr.surg[(pearson_corr.surg >= 0.3) | (pearson_corr.surg <= -0.3)].iteritems():
    if key != "surg":
        print("Pearson correlation coefficient with {} is: {:.2f}.".format(key, value))

print("")
print("SPEARMAN CORRELATIONS")

for key, value in spearman_corr.surg[(spearman_corr.surg >= 0.3) | (spearman_corr.surg <= -0.3)].iteritems():
    if key != "surg":
        print("Spearman correlation coefficient with {} is: {:.2f}.".format(key, value))

print("")
print("KENDALL CORRELATIONS")

for key, value in kendall_corr.surg[(kendall_corr.surg >= 0.3) | (kendall_corr.surg <= -0.3)].iteritems():
    if key != "surg":
        print("Kendall correlation coefficient with {} is: {:.2f}.".format(key, value))

# %%  
fl = [["ct"], ["cts"], ["pts"], ["ptmm"],
      ["ct", "cts"], ["cts", "pts"], ["ct", "ptmm"],
      ["ct", "pts", "ptmm"],
      ["ct", "cts", "pts", "ptmm"]]

surg_scores = {}

for f in fl:
    
    for i in range(1, 4):
        df_train, df_test = train_test_split(df, test_size = 0.1)
        df_train, df_validation = train_test_split(df_train, test_size = 0.11113)
        
        #Initialize a SimpleImputer model
        imputer = datawig.SimpleImputer(
            input_columns=f,
            output_column="surg",
            output_path ="../imputers/imputer_model"
        )

        #Fit an imputer model on the train data
        imputer.fit(train_df = df_train, test_df = df_validation, num_epochs = 3)

        #Impute missing values and return original dataframe with predictions
        imputed = imputer.predict(df_test)

        not_empty_df_test = imputed.loc[imputed.surg.notnull()]

        not_empty_df_test.surg = not_empty_df_test.surg.map(inv_categorical_dict)
        not_empty_df_test.surg_imputed = not_empty_df_test.surg_imputed.map(inv_categorical_dict)

        acc = accuracy_score(not_empty_df_test.surg, not_empty_df_test.surg_imputed)
        avg_prec = precision_score(not_empty_df_test.surg, not_empty_df_test.surg_imputed, average="macro")
        weight_prec = precision_score(not_empty_df_test.surg, not_empty_df_test.surg_imputed, average="weighted")
        avg_rec = recall_score(not_empty_df_test.surg, not_empty_df_test.surg_imputed, average="macro")
        weight_rec = recall_score(not_empty_df_test.surg, not_empty_df_test.surg_imputed, average="weighted")
        avg_f1 = f1_score(not_empty_df_test.surg, not_empty_df_test.surg_imputed, average="macro")
        weight_f1 = f1_score(not_empty_df_test.surg, not_empty_df_test.surg_imputed, average="weighted")

        metrics = [acc, avg_prec, weight_prec, avg_rec, weight_rec, avg_f1, weight_f1]
        key = '_'.join(f) + "_cv_" + str(i)

        surg_scores[key] = metrics

# %%  
surg_scores_matrix = pd.DataFrame(surg_scores)
surg_scores_matrix.index = cat_indices
surg_scores_matrix = surg_scores_matrix.T
surg_scores_matrix

# %%  
surg_scores_matrix.to_excel("../stats/imputation/surgery_scores_matrix_{}.xlsx".format(date_notation))

# %%
# Lymph nodes features (rly and ply) imputation

print("PEARSON RLY CORRELATIONS")
for key, value in pearson_corr.rly[(pearson_corr.rly >= 0.3) | (pearson_corr.rly <= -0.3)].iteritems():
    if key != "rly":
        print("Pearson correlation coefficient with {} is: {:.2f}.".format(key, value))

print("")
print("SPEARMAN RLY CORRELATIONS")

for key, value in spearman_corr.rly[(spearman_corr.rly >= 0.3) | (spearman_corr.rly <= -0.3)].iteritems():
    if key != "rly":
        print("Spearman correlation coefficient with {} is: {:.2f}.".format(key, value))

print("")
print("KENDALL RLY CORRELATIONS")

for key, value in kendall_corr.rly[(kendall_corr.rly >= 0.3) | (kendall_corr.rly <= -0.3)].iteritems():
    if key != "rly":
        print("Kendall correlation coefficient with {} is: {:.2f}.".format(key, value))

# %%
print("PEARSON PLY CORRELATIONS")
for key, value in pearson_corr.ply[(pearson_corr.ply >= 0.3) | (pearson_corr.ply <= -0.3)].iteritems():
    if key != "ply":
        print("Pearson correlation coefficient with {} is: {:.2f}.".format(key, value))

print("")
print("SPEARMAN PLY CORRELATIONS")

for key, value in spearman_corr.ply[(spearman_corr.ply >= 0.3) | (spearman_corr.ply <= -0.3)].iteritems():
    if key != "ply":
        print("Spearman correlation coefficient with {} is: {:.2f}.".format(key, value))

print("")
print("KENDALL PLY CORRELATIONS")

for key, value in kendall_corr.ply[(kendall_corr.ply >= 0.3) | (kendall_corr.ply <= -0.3)].iteritems():
    if key != "ply":
        print("Kendall correlation coefficient with {} is: {:.2f}.".format(key, value))

# %%  
fl = [["cn"], ["pn"], ["pts"], 
      ["cn", "pn"], ["cn", "pts"], ["pn", "pts"], 
      ["cn", "pn", "pts"]]

rly_scores = {}

for f in fl:
    
    for i in range(1, 4):
        df_train, df_test = train_test_split(df, test_size = 0.1)
        df_train, df_validation = train_test_split(df_train, test_size = 0.11113)
        
        #Initialize a SimpleImputer model
        imputer = datawig.SimpleImputer(
            input_columns=f,
            output_column="rly",
            output_path ="../imputers/imputer_model"
        )

        #Fit an imputer model on the train data
        imputer.fit(train_df = df_train, test_df = df_validation, num_epochs = 3)

        #Impute missing values and return original dataframe with predictions
        imputed = imputer.predict(df_test)

        not_empty_df_test = imputed.loc[imputed.rly.notnull()]

        mse = mean_squared_error(not_empty_df_test.rly, not_empty_df_test.rly_imputed)
        mae = mean_absolute_error(not_empty_df_test.rly, not_empty_df_test.rly_imputed)
        evs = explained_variance_score(not_empty_df_test.rly, not_empty_df_test.rly_imputed)
        r2 = r2_score(not_empty_df_test.rly, not_empty_df_test.rly_imputed)

        metrics = [mse, mae, evs, r2]
        key = '_'.join(f)  + "_cv_" + str(i)

        rly_scores[key] = metrics

# %%  
rly_scores_matrix = pd.DataFrame(rly_scores)
rly_scores_matrix.index = cont_indices
rly_scores_matrix = rly_scores_matrix.T
rly_scores_matrix

# %%  
rly_scores_matrix.to_excel("../stats/imputation/rly_scores_matrix_{}.xlsx".format(date_notation)) 

# %%  
rly_imputer_columns = ["pn", "pts"]

# %%  
fl = [["cn"], ["pn"], ["pts"], ["rly"], 
      ["pn", "rly"],
      ["cn", "pn", "rly"], ["pn", "pts", "rly"], ["cn", "pn", "pts"],
      ["cn", "pn", "pts", "rly"]]

ply_scores = {}

for f in fl:
    
    for i in range(1, 4):
        df_train, df_test = train_test_split(df, test_size = 0.1)
        df_train, df_validation = train_test_split(df_train, test_size = 0.11113)

        #Initialize a SimpleImputer model
        imputer = datawig.SimpleImputer(
            input_columns=f,
            output_column="ply",
            output_path ="../imputers/imputer_model"
        )

        #Fit an imputer model on the train data
        imputer.fit(train_df = df_train, test_df = df_validation, num_epochs = 3)

        #Impute missing values and return original dataframe with predictions
        imputed = imputer.predict(df_test)

        not_empty_df_test = imputed.loc[imputed.ply.notnull()]

        mse = mean_squared_error(not_empty_df_test.ply, not_empty_df_test.ply_imputed)
        mae = mean_absolute_error(not_empty_df_test.ply, not_empty_df_test.ply_imputed)
        evs = explained_variance_score(not_empty_df_test.ply, not_empty_df_test.ply_imputed)
        r2 = r2_score(not_empty_df_test.ply, not_empty_df_test.ply_imputed)

        metrics = [mse, mae, evs, r2]
        key = '_'.join(f)  + "_cv_" + str(i)

        ply_scores[key] = metrics

# %%  
ply_scores_matrix = pd.DataFrame(ply_scores)
ply_scores_matrix.index = cont_indices
ply_scores_matrix = ply_scores_matrix.T
ply_scores_matrix

# %%  
ply_scores_matrix.to_excel("../stats/imputation/ply_scores_matrix_{}.xlsx".format(date_notation))

# %%  
ply_imputer_columns = ["pn", "pts", "rly"]

# %%
# Tumour grade (grd) imputation

print("PEARSON GRD CORRELATIONS")
for key, value in pearson_corr.grd[(pearson_corr.grd >= 0.3) | (pearson_corr.grd <= -0.3)].iteritems():
    if key != "grd":
        print("Pearson correlation coefficient with {} is: {:.2f}.".format(key, value))

print("")
print("SPEARMAN GRD CORRELATIONS")

for key, value in spearman_corr.grd[(spearman_corr.grd >= 0.3) | (spearman_corr.grd <= -0.3)].iteritems():
    if key != "grd":
        print("Spearman correlation coefficient with {} is: {:.2f}.".format(key, value))

print("")
print("KENDALL GRD CORRELATIONS")

for key, value in kendall_corr.grd[(kendall_corr.grd >= 0.3) | (kendall_corr.grd <= -0.3)].iteritems():
    if key != "grd":
        print("Kendall correlation coefficient with {} is: {:.2f}.".format(key, value))

# %%  
fl = [["ER"], ["PR"], 
      ["ER", "PR"]]

grd_scores = {}

for f in fl:
    
    for i in range(1, 4):
        df_train, df_test = train_test_split(df, test_size = 0.1)
        df_train, df_validation = train_test_split(df_train, test_size = 0.11113)
        
        #Initialize a SimpleImputer model
        imputer = datawig.SimpleImputer(
            input_columns=f,
            output_column="grd",
            output_path ="../imputers/imputer_model"
        )

        #Fit an imputer model on the train data
        imputer.fit(train_df = df_train, test_df = df_validation, num_epochs = 3)

        #Impute missing values and return original dataframe with predictions
        imputed = imputer.predict(df_test)

        not_empty_df_test = imputed.loc[imputed.grd.notnull()]

        not_empty_df_test.grd = not_empty_df_test.grd.map(inv_categorical_dict)
        not_empty_df_test.grd_imputed = not_empty_df_test.grd_imputed.map(inv_categorical_dict)

        acc = accuracy_score(not_empty_df_test.grd, not_empty_df_test.grd_imputed)
        avg_prec = precision_score(not_empty_df_test.grd, not_empty_df_test.grd_imputed, average="macro")
        weight_prec = precision_score(not_empty_df_test.grd, not_empty_df_test.grd_imputed, average="weighted")
        avg_rec = recall_score(not_empty_df_test.grd, not_empty_df_test.grd_imputed, average="macro")
        weight_rec = recall_score(not_empty_df_test.grd, not_empty_df_test.grd_imputed, average="weighted")
        avg_f1 = f1_score(not_empty_df_test.grd, not_empty_df_test.grd_imputed, average="macro")
        weight_f1 = f1_score(not_empty_df_test.grd, not_empty_df_test.grd_imputed, average="weighted")

        metrics = [acc, avg_prec, weight_prec, avg_rec, weight_rec, avg_f1, weight_f1]
        key = '_'.join(f) + "_cv_" + str(i)

        grd_scores[key] = metrics

# %%  
grd_scores_matrix = pd.DataFrame(grd_scores)
grd_scores_matrix.index = cat_indices
grd_scores_matrix = grd_scores_matrix.T
grd_scores_matrix

# %%  
grd_scores_matrix.to_excel("../stats/imputation/grd_scores_matrix_{}.xlsx".format(date_notation))

# %%
# TNM staging imputation

print("PEARSON CT CORRELATIONS")
for key, value in pearson_corr.ct[(pearson_corr.ct >= 0.3) | (pearson_corr.ct <= -0.3)].iteritems():
    if key != "ct":
        print("Pearson correlation coefficient with {} is: {:.2f}.".format(key, value))

print("")
print("SPEARMAN CT CORRELATIONS")

for key, value in spearman_corr.ct[(spearman_corr.ct >= 0.3) | (spearman_corr.ct <= -0.3)].iteritems():
    if key != "ct":
        print("Spearman correlation coefficient with {} is: {:.2f}.".format(key, value))

print("")
print("KENDALL CT CORRELATIONS")

for key, value in kendall_corr.ct[(kendall_corr.ct >= 0.3) | (kendall_corr.ct <= -0.3)].iteritems():
    if key != "ct":
        print("Kendall correlation coefficient with {} is: {:.2f}.".format(key, value))

# %%  
fl = [["pt"], ["cts"], ["pts"],
      ["pt", "cts"], ["pt", "pts"], ["cts", "pts"],
      ["pt", "cts", "pts"],
      ["pt", "cts", "pts", "ptmm", "surg"]]

ct_scores = {}

for f in fl:
    
    for i in range(1, 4):
        df_train, df_test = train_test_split(df, test_size = 0.1)
        df_train, df_validation = train_test_split(df_train, test_size = 0.11113)
        
        #Initialize a SimpleImputer model
        imputer = datawig.SimpleImputer(
            input_columns=f,
            output_column="ct",
            output_path ="../imputers/imputer_model"
        )

        #Fit an imputer model on the train data
        imputer.fit(train_df = df_train, test_df = df_validation, num_epochs = 3)

        #Impute missing values and return original dataframe with predictions
        imputed = imputer.predict(df_test)

        not_empty_df_test = imputed.loc[imputed.ct.notnull()]

        not_empty_df_test.ct = not_empty_df_test.ct.map(inv_categorical_dict)
        not_empty_df_test.ct_imputed = not_empty_df_test.ct_imputed.map(inv_categorical_dict)

        acc = accuracy_score(not_empty_df_test.ct, not_empty_df_test.ct_imputed)
        avg_prec = precision_score(not_empty_df_test.ct, not_empty_df_test.ct_imputed, average="macro")
        weight_prec = precision_score(not_empty_df_test.ct, not_empty_df_test.ct_imputed, average="weighted")
        avg_rec = recall_score(not_empty_df_test.ct, not_empty_df_test.ct_imputed, average="macro")
        weight_rec = recall_score(not_empty_df_test.ct, not_empty_df_test.ct_imputed, average="weighted")
        avg_f1 = f1_score(not_empty_df_test.ct, not_empty_df_test.ct_imputed, average="macro")
        weight_f1 = f1_score(not_empty_df_test.ct, not_empty_df_test.ct_imputed, average="weighted")

        metrics = [acc, avg_prec, weight_prec, avg_rec, weight_rec, avg_f1, weight_f1]
        key = '_'.join(f) + "_cv_" + str(i)
    
        ct_scores[key] = metrics

# %%  
ct_scores_matrix = pd.DataFrame(ct_scores)
ct_scores_matrix.index = cat_indices
ct_scores_matrix = ct_scores_matrix.T
ct_scores_matrix

# %%  
ct_scores_matrix.to_excel("../stats/imputation/ct_scores_matrix_{}.xlsx".format(date_notation))

# %%  
ct_imputer_columns = ["pt", "cts", "pts", "ptmm", "surg"]

# %%
print("PEARSON CN CORRELATIONS")
for key, value in pearson_corr.cn[(pearson_corr.cn >= 0.3) | (pearson_corr.cn <= -0.3)].iteritems():
    if key != "cn":
        print("Pearson correlation coefficient with {} is: {:.2f}.".format(key, value))

print("")
print("SPEARMAN CN CORRELATIONS")

for key, value in spearman_corr.cn[(spearman_corr.cn >= 0.3) | (spearman_corr.cn <= -0.3)].iteritems():
    if key != "cn":
        print("Spearman correlation coefficient with {} is: {:.2f}.".format(key, value))

print("")
print("KENDALL CN CORRELATIONS")

for key, value in kendall_corr.cn[(kendall_corr.cn >= 0.3) | (kendall_corr.cn <= -0.3)].iteritems():
    if key != "cn":
        print("Kendall correlation coefficient with {} is: {:.2f}.".format(key, value))

# %%  
fl = [["cts"], ["pts"], ["ply"],
      ["cts", "ply"], ["cts", "pts"], ["pts", "ply"],
      ["cts", "pts", "ply"],
      ["pn", "cts", "pts", "rly", "ply", "neoct"]]

cn_scores = {}

for f in fl:
    
    for i in range(1, 4):
        df_train, df_test = train_test_split(df, test_size = 0.1)
        df_train, df_validation = train_test_split(df_train, test_size = 0.11113)
        
        #Initialize a SimpleImputer model
        imputer = datawig.SimpleImputer(
            input_columns=f,
            output_column="cn",
            output_path ="../imputers/imputer_model"
        )

        #Fit an imputer model on the train data
        imputer.fit(train_df = df_train, test_df = df_validation, num_epochs = 3)

        #Impute missing values and return original dataframe with predictions
        imputed = imputer.predict(df_test)

        not_empty_df_test = imputed.loc[imputed.cn.notnull()]

        not_empty_df_test.cn = not_empty_df_test.cn.map(inv_categorical_dict)
        not_empty_df_test.cn_imputed = not_empty_df_test.cn_imputed.map(inv_categorical_dict)

        acc = accuracy_score(not_empty_df_test.cn, not_empty_df_test.cn_imputed)
        avg_prec = precision_score(not_empty_df_test.cn, not_empty_df_test.cn_imputed, average="macro")
        weight_prec = precision_score(not_empty_df_test.cn, not_empty_df_test.cn_imputed, average="weighted")
        avg_rec = recall_score(not_empty_df_test.cn, not_empty_df_test.cn_imputed, average="macro")
        weight_rec = recall_score(not_empty_df_test.cn, not_empty_df_test.cn_imputed, average="weighted")
        avg_f1 = f1_score(not_empty_df_test.cn, not_empty_df_test.cn_imputed, average="macro")
        weight_f1 = f1_score(not_empty_df_test.cn, not_empty_df_test.cn_imputed, average="weighted")

        metrics = [acc, avg_prec, weight_prec, avg_rec, weight_rec, avg_f1, weight_f1]
        key = '_'.join(f) + "_cv_" + str(i)
    
        cn_scores[key] = metrics

# %%  
cn_scores_matrix = pd.DataFrame(cn_scores)
cn_scores_matrix.index = cat_indices
cn_scores_matrix = cn_scores_matrix.T
cn_scores_matrix

# %%  
cn_scores_matrix.to_excel("../stats/imputation/cn_scores_matrix_{}.xlsx".format(date_notation))

# %%  
cn_imputer_columns = ["pn", "cts", "pts", "rly", "ply", "neoct"]

# %%
print("PEARSON CTS CORRELATIONS")
for key, value in pearson_corr.cts[(pearson_corr.cts >= 0.3) | (pearson_corr.cts <= -0.3)].iteritems():
    if key != "cts":
        print("Pearson correlation coefficient with {} is: {:.2f}.".format(key, value))

print("")
print("SPEARMAN CTS CORRELATIONS")

for key, value in spearman_corr.cts[(spearman_corr.cts >= 0.3) | (spearman_corr.cts <= -0.3)].iteritems():
    if key != "cts":
        print("Spearman correlation coefficient with {} is: {:.2f}.".format(key, value))

print("")
print("KENDALL CTS CORRELATIONS")

for key, value in kendall_corr.cts[(kendall_corr.cts >= 0.3) | (kendall_corr.cts <= -0.3)].iteritems():
    if key != "cts":
        print("Kendall correlation coefficient with {} is: {:.2f}.".format(key, value))

# %%  
fl = [["ct"], ["cn"], ["pts"], 
      ["ct", "cn"], ["ct", "pts"], ["cn", "pts"],
      ["ct", "pt", "pts"],
      ["ct", "cn", "pt", "pn", "pts", "ply", "ptmm", "neoct"]]

cts_scores = {}

for f in fl:
    
    for i in range(1, 4):
        df_train, df_test = train_test_split(df, test_size = 0.1)
        df_train, df_validation = train_test_split(df_train, test_size = 0.11113)
        
        #Initialize a SimpleImputer model
        imputer = datawig.SimpleImputer(
            input_columns=f,
            output_column="cts",
            output_path ="../imputers/imputer_model"
        )

        #Fit an imputer model on the train data
        imputer.fit(train_df = df_train, test_df = df_validation, num_epochs = 3)

        #Impute missing values and return original dataframe with predictions
        imputed = imputer.predict(df_test)

        not_empty_df_test = imputed.loc[imputed.cts.notnull()]

        not_empty_df_test.cts = not_empty_df_test.cts.map(inv_categorical_dict)
        not_empty_df_test.cts_imputed = not_empty_df_test.cts_imputed.map(inv_categorical_dict)

        acc = accuracy_score(not_empty_df_test.cts, not_empty_df_test.cts_imputed)
        avg_prec = precision_score(not_empty_df_test.cts, not_empty_df_test.cts_imputed, average="macro")
        weight_prec = precision_score(not_empty_df_test.cts, not_empty_df_test.cts_imputed, average="weighted")
        avg_rec = recall_score(not_empty_df_test.cts, not_empty_df_test.cts_imputed, average="macro")
        weight_rec = recall_score(not_empty_df_test.cts, not_empty_df_test.cts_imputed, average="weighted")
        avg_f1 = f1_score(not_empty_df_test.cts, not_empty_df_test.cts_imputed, average="macro")
        weight_f1 = f1_score(not_empty_df_test.cts, not_empty_df_test.cts_imputed, average="weighted")

        metrics = [acc, avg_prec, weight_prec, avg_rec, weight_rec, avg_f1, weight_f1]
        key = '_'.join(f) + "_cv_" + str(i)
    
        cts_scores[key] = metrics

# %%  
cts_scores_matrix = pd.DataFrame(cts_scores)
cts_scores_matrix.index = cat_indices
cts_scores_matrix = cts_scores_matrix.T
cts_scores_matrix

# %%  
cts_scores_matrix.to_excel("../stats/imputation/cts_scores_matrix_{}.xlsx".format(date_notation))

# %%  
cts_imputer_columns = ["ct", "cn", "pt", "pn", "ply", "pts", "ptmm", "neoct"]

# %%
print("PEARSON PT CORRELATIONS")
for key, value in pearson_corr.pt[(pearson_corr.pt >= 0.3) | (pearson_corr.pt <= -0.3)].iteritems():
    if key != "pt":
        print("Pearson correlation coefficient with {} is: {:.2f}.".format(key, value))

print("")
print("SPEARMAN PT CORRELATIONS")

for key, value in spearman_corr.pt[(spearman_corr.pt >= 0.3) | (spearman_corr.pt <= -0.3)].iteritems():
    if key != "pt":
        print("Spearman correlation coefficient with {} is: {:.2f}.".format(key, value))

print("")
print("KENDALL PT CORRELATIONS")

for key, value in kendall_corr.pt[(kendall_corr.pt >= 0.3) | (kendall_corr.pt <= -0.3)].iteritems():
    if key != "pt":
        print("Kendall correlation coefficient with {} is: {:.2f}.".format(key, value))

# %%  
fl = [["ct"], ["pts"], ["ptmm"], 
      ["ct", "pts"], ["ct", "ptmm"], ["pts", "ptmm"],
      ["ct", "pts", "ptmm"],
      ["ct", "cts", "pts", "ptmm", "ply"]]

pt_scores = {}

for f in fl:
    
    for i in range(1, 4):
        df_train, df_test = train_test_split(df, test_size = 0.1)
        df_train, df_validation = train_test_split(df_train, test_size = 0.11113)
        
        #Initialize a SimpleImputer model
        imputer = datawig.SimpleImputer(
            input_columns=f,
            output_column="pt",
            output_path ="../imputers/imputer_model"
        )

        #Fit an imputer model on the train data
        imputer.fit(train_df = df_train, test_df = df_validation, num_epochs = 3)

        #Impute missing values and return original dataframe with predictions
        imputed = imputer.predict(df_test)

        not_empty_df_test = imputed.loc[imputed.pt.notnull()]

        not_empty_df_test.pt = not_empty_df_test.pt.map(inv_categorical_dict)
        not_empty_df_test.pt_imputed = not_empty_df_test.pt_imputed.map(inv_categorical_dict)

        acc = accuracy_score(not_empty_df_test.pt, not_empty_df_test.pt_imputed)
        avg_prec = precision_score(not_empty_df_test.pt, not_empty_df_test.pt_imputed, average="macro")
        weight_prec = precision_score(not_empty_df_test.pt, not_empty_df_test.pt_imputed, average="weighted")
        avg_rec = recall_score(not_empty_df_test.pt, not_empty_df_test.pt_imputed, average="macro")
        weight_rec = recall_score(not_empty_df_test.pt, not_empty_df_test.pt_imputed, average="weighted")
        avg_f1 = f1_score(not_empty_df_test.pt, not_empty_df_test.pt_imputed, average="macro")
        weight_f1 = f1_score(not_empty_df_test.pt, not_empty_df_test.pt_imputed, average="weighted")

        metrics = [acc, avg_prec, weight_prec, avg_rec, weight_rec, avg_f1, weight_f1]
        key = '_'.join(f) + "_cv_" + str(i)
    
        pt_scores[key] = metrics

# %%  
pt_scores_matrix = pd.DataFrame(pt_scores)
pt_scores_matrix.index = cat_indices
pt_scores_matrix = pt_scores_matrix.T
pt_scores_matrix

# %%  
pt_scores_matrix.to_excel("../stats/imputation/pt_scores_matrix_{}.xlsx".format(date_notation))

# %%  
pt_imputer_columns = ["ct", "cts", "pts", "ply", "ptmm"]

# %%
print("PEARSON PN CORRELATIONS")
for key, value in pearson_corr.pn[(pearson_corr.pn >= 0.3) | (pearson_corr.pn <= -0.3)].iteritems():
    if key != "pn":
        print("Pearson correlation coefficient with {} is: {:.2f}.".format(key, value))

print("")
print("SPEARMAN PN CORRELATIONS")

for key, value in spearman_corr.pn[(spearman_corr.pn >= 0.3) | (spearman_corr.pn <= -0.3)].iteritems():
    if key != "pn":
        print("Spearman correlation coefficient with {} is: {:.2f}.".format(key, value))

print("")
print("KENDALL PN CORRELATIONS")

for key, value in kendall_corr.pt[(kendall_corr.pn >= 0.3) | (kendall_corr.pn <= -0.3)].iteritems():
    if key != "pn":
        print("Kendall correlation coefficient with {} is: {:.2f}.".format(key, value))

# %%  
fl = [["pts"], ["ply"], 
      ["pts", "ply"],
      ["cn", "cts", "pts", "rly", "ply"]]

pn_scores = {}

for f in fl:
    
    for i in range(1, 4):
        df_train, df_test = train_test_split(df, test_size = 0.1)
        df_train, df_validation = train_test_split(df_train, test_size = 0.11113)
        
        #Initialize a SimpleImputer model
        imputer = datawig.SimpleImputer(
            input_columns=f,
            output_column="pn",
            output_path ="../imputers/imputer_model"
        )

        #Fit an imputer model on the train data
        imputer.fit(train_df = df_train, test_df = df_validation, num_epochs = 3)

        #Impute missing values and return original dataframe with predictions
        imputed = imputer.predict(df_test)

        not_empty_df_test = imputed.loc[imputed.pn.notnull()]

        not_empty_df_test.pn = not_empty_df_test.pn.map(inv_categorical_dict)
        not_empty_df_test.pn_imputed = not_empty_df_test.pn_imputed.map(inv_categorical_dict)

        acc = accuracy_score(not_empty_df_test.pn, not_empty_df_test.pn_imputed)
        avg_prec = precision_score(not_empty_df_test.pn, not_empty_df_test.pn_imputed, average="macro")
        weight_prec = precision_score(not_empty_df_test.pn, not_empty_df_test.pn_imputed, average="weighted")
        avg_rec = recall_score(not_empty_df_test.pn, not_empty_df_test.pn_imputed, average="macro")
        weight_rec = recall_score(not_empty_df_test.pn, not_empty_df_test.pn_imputed, average="weighted")
        avg_f1 = f1_score(not_empty_df_test.pn, not_empty_df_test.pn_imputed, average="macro")
        weight_f1 = f1_score(not_empty_df_test.pn, not_empty_df_test.pn_imputed, average="weighted")

        metrics = [acc, avg_prec, weight_prec, avg_rec, weight_rec, avg_f1, weight_f1]
        key = '_'.join(f) + "_cv_" + str(i)
    
        pn_scores[key] = metrics

# %%  
pn_scores_matrix = pd.DataFrame(pn_scores)
pn_scores_matrix.index = cat_indices
pn_scores_matrix = pn_scores_matrix.T
pn_scores_matrix

# %%  
pn_scores_matrix.to_excel("../stats/imputation/pn_scores_matrix_{}.xlsx".format(date_notation))

# %%  
pn_imputer_columns = ["cn", "cts", "pts", "rly", "ply"]

# %%
# Actual imputation

# %%  
df_train = df.loc[df.men.notnull() & df.age.notnull()]

#Initialize a SimpleImputer model
men_imputer = datawig.SimpleImputer(
    input_columns = men_imputer_columns,
    output_column = "men",
    output_path = "../imputers/men_imputer_model_{}".format(date_notation)
)

#Fit an imputer model on the train data
men_imputer.fit(train_df = df_train, num_epochs = 10)

# %%  
df_train = df.loc[df.ptmm.notnull() & df.pt.notnull()]

#Initialize a SimpleImputer model
ptmm_imputer = datawig.SimpleImputer(
    input_columns = ptmm_imputer_columns,
    output_column = "ptmm",
    output_path = "../imputers/ptmm_imputer_model_{}".format(date_notation)
)

#Fit an imputer model on the train data
ptmm_imputer.fit(train_df = df_train, num_epochs = 10)

# %%  
df_train = df.loc[df.PR.notnull() & df.ER.notnull() & df.grd.notnull()]

#Initialize a SimpleImputer model
ER_imputer = datawig.SimpleImputer(
    input_columns=["PR", "grd"],
    output_column="ER",
    output_path ="../imputers/ER_imputer_model_{}".format(date_notation)
)

#Fit an imputer model on the train data
ER_imputer.fit(train_df = df_train, num_epochs = 10)

# %%  
df_train = df.loc[df.PR.notnull() & df.ER.notnull()]

#Initialize a SimpleImputer model
PR_imputer = datawig.SimpleImputer(
    input_columns=PR_imputer_columns,
    output_column="PR",
    output_path ="../imputers/PR_imputer_model_{}".format(date_notation)
)

#Fit an imputer model on the train data
PR_imputer.fit(train_df = df_train, num_epochs = 10)

# %%  
df_train = df.loc[df.pn.notnull() & df.rly.notnull()]

#Initialize a SimpleImputer model
rly_imputer = datawig.SimpleImputer(
    input_columns=rly_imputer_columns,
    output_column="rly",
    output_path ="../imputers/rly_imputer_model_{}".format(date_notation)
)

#Fit an imputer model on the train data
rly_imputer.fit(train_df = df_train, num_epochs = 10)

# %%  
df_train = df.loc[df.ply.notnull() & df.pn.notnull() & df.rly.notnull()]

#Initialize a SimpleImputer model
ply_imputer = datawig.SimpleImputer(
    input_columns=ply_imputer_columns,
    output_column="ply",
    output_path ="../imputers/ply_imputer_model_{}".format(date_notation)
)

#Fit an imputer model on the train data
ply_imputer.fit(train_df = df_train, num_epochs = 10)

# %%  
df_train = df.loc[df.cts.notnull() & df.pt.notnull() & df.ct.notnull()]

#Initialize a SimpleImputer model
ct_imputer = datawig.SimpleImputer(
    input_columns=ct_imputer_columns,
    output_column="ct",
    output_path ="../imputers/ct_imputer_model_{}".format(date_notation)
)

#Fit an imputer model on the train data
ct_imputer.fit(train_df = df_train, num_epochs = 10)

# %%  
df_train = df.loc[df.cn.notnull() & df.cts.notnull()]

#Initialize a SimpleImputer model
cn_imputer = datawig.SimpleImputer(
    input_columns=cn_imputer_columns,
    output_column="cn",
    output_path ="../imputers/cn_imputer_model_{}".format(date_notation)
)

#Fit an imputer model on the train data
cn_imputer.fit(train_df = df_train, num_epochs = 10)

# %%  
df_train = df.loc[df.cts.notnull() & df.ct.notnull() & df.pts.notnull()]

#Initialize a SimpleImputer model
cts_imputer = datawig.SimpleImputer(
    input_columns=cts_imputer_columns,
    output_column="cts",
    output_path ="../imputers/cts_imputer_model_{}".format(date_notation)
)

#Fit an imputer model on the train data
cts_imputer.fit(train_df = df_train, num_epochs = 10)

# %%  
df_train = df.loc[df.pt.notnull() & df.ptmm.notnull()]

#Initialize a SimpleImputer model
pt_imputer = datawig.SimpleImputer(
    input_columns=pt_imputer_columns,
    output_column="pt",
    output_path ="../imputers/pt_imputer_model_{}".format(date_notation)
)

#Fit an imputer model on the train data
pt_imputer.fit(train_df = df_train, num_epochs = 10)

# %%  
df_train = df.loc[df.pn.notnull() & df.cn.notnull() & df.cts.notnull() & df.pts.notnull() & df.rly.notnull() & df.ply.notnull()]

#Initialize a SimpleImputer model
pn_imputer = datawig.SimpleImputer(
    input_columns=pn_imputer_columns,
    output_column="pn",
    output_path ="../imputers/pn_imputer_model_{}".format(date_notation)
)

#Fit an imputer model on the train data
pn_imputer.fit(train_df = df_train, num_epochs = 10)

# %%  
# Load models that were saved from above training and testing iterations if needed after kernel restart

men_imputer = datawig.SimpleImputer.load("../imputers/men_imputer_model_05Jul")
ptmm_imputer = datawig.SimpleImputer.load("../imputers/ptmm_imputer_model_05Jul")
ER_imputer = datawig.SimpleImputer.load("../imputers/ER_imputer_model_05Jul")
PR_imputer = datawig.SimpleImputer.load("../imputers/PR_imputer_model_05Jul")
rly_imputer = datawig.SimpleImputer.load("../imputers/rly_imputer_model_05Jul")
ply_imputer = datawig.SimpleImputer.load("../imputers/ply_imputer_model_05Jul")
ct_imputer = datawig.SimpleImputer.load("../imputers/ct_imputer_model_05Jul") 
cn_imputer = datawig.SimpleImputer.load("../imputers/cn_imputer_model_05Jul")
cts_imputer = datawig.SimpleImputer.load("../imputers/cts_imputer_model_05Jul")
pt_imputer = datawig.SimpleImputer.load("../imputers/pt_imputer_model_05Jul")
pn_imputer = datawig.SimpleImputer.load("../imputers/pn_imputer_model_05Jul")

# %%  
# Impute all missing values with the trained deep learning networks

# men
df_men_imputed = men_imputer.predict(df.loc[df.men.isnull()])
df.loc[df.men.isnull(), "men"] = df_men_imputed.men_imputed

# ptmm
df_ptmm_imputed = ptmm_imputer.predict(df.loc[df.ptmm.isnull() & df.pt.notnull()])
df.loc[(df.ptmm.isnull() & df.pt.notnull()), "ptmm"] = df_ptmm_imputed.ptmm_imputed

# ER
df_ER_imputed = ER_imputer.predict(df.loc[df.ER.isnull() & df.PR.notnull() & df.grd.notnull()])
df.loc[(df.ER.isnull() & df.PR.notnull() & df.grd.notnull()), "ER"] = df_ER_imputed.ER_imputed

# PR
df_PR_imputed = PR_imputer.predict(df.loc[df.PR.isnull() & df.ER.notnull()])
df.loc[(df.PR.isnull() & df.ER.notnull()), "PR"] = df_PR_imputed.PR_imputed

# rly
df_rly_imputed = rly_imputer.predict(df.loc[df.rly.isnull() & df.pn.notnull()])
df.loc[(df.rly.isnull() & df.pn.notnull()), "rly"] = df_rly_imputed.rly_imputed.round()

# ply
df_ply_imputed = ply_imputer.predict(df.loc[df.ply.isnull() & df.pn.notnull() & df.rly.notnull()])
df.loc[df.ply.isnull() & df.pn.notnull() & df.rly.notnull(), "ply"] = df_ply_imputed.ply_imputed.round()

# ct
df_ct_imputed = ct_imputer.predict(df.loc[df.ct.isnull() & df.cts.notnull() & df.pt.notnull() 
                                          & df.cn.notnull() & df.ptmm.notnull()])
df.loc[df.ct.isnull() & df.cts.notnull() & df.pt.notnull() & df.cn.notnull() 
       & df.ptmm.notnull(), "ct"] = df_ct_imputed.ct_imputed

# cn
df_cn_imputed = cn_imputer.predict(df.loc[df.cn.isnull() & df.cts.notnull()])
df.loc[df.cn.isnull() & df.cts.notnull(), "cn"] = df_cn_imputed.cn_imputed

# cts
df_cts_imputed = cts_imputer.predict(df.loc[df.cts.isnull() & df.ct.notnull() & df.pts.notnull()])
df.loc[df.cts.isnull() & df.ct.notnull() & df.pts.notnull(), "cts"] = df_cts_imputed.cts_imputed

# pt
df_pt_imputed = pt_imputer.predict(df.loc[df.pt.isnull() & df.ptmm.notnull()])
df.loc[df.pt.isnull() & df.ptmm.notnull(), "pt"] = df_pt_imputed.pt_imputed

# pn
df_pn_imputed = pn_imputer.predict(df.loc[df.pn.isnull() & df.cn.notnull() & df.cts.notnull() 
                                          & df.pts.notnull() & df.rly.notnull() & df.ply.notnull()])
df.loc[df.pn.isnull() & df.cn.notnull() & df.cts.notnull() & df.pts.notnull() & df.rly.notnull() 
       & df.ply.notnull(), "pn"] = df_pn_imputed.pn_imputed

# %%  
print("As mentioned earlier, a full dataframe would contain {} values, and before any \
imputation there was a total of 7.18% of missing values.".format(all_values))

missing_values = df.isna().sum().sum()

print("After deep learning imputation, the dataframe contains {} missing values, which \
is now {:.2f}% of all possible values.".format(missing_values, (missing_values/all_values)*100))

# %%  
# Imputed Data Export

for feature in categorical_features:
    df[feature] = df[feature].map(inv_categorical_dict)

# %%  
df.sample(7)

# %%
# Specify a list of features that are categorical and have to be dummified for processing by kNN
features_to_dummify = ["loc", "lat", "mor", "grd", "diag", "ct", "cn", "pt", "pn", "cts", "pts", "fh", 
                       "trh", "surg", "HER2", "men"]

# Define a feature to impute values with a manually developed kNN algorithm
def manual_knn_impute(df):
    # Retrieve a list of which columns contain missing values
    cols_with_na = df.columns[df.isna().any()].tolist()
    cols_with_na = [col for col in cols_with_na if col not in ["ratly", "horm", "trneg"]]

    for col in cols_with_na:
        if col in features_to_dummify:
            features_to_dummify_temp = features_to_dummify.remove(col)
            df_dummified = pd.get_dummies(df, prefix=features_to_dummify_temp, 
                                          columns=features_to_dummify_temp, dummy_na=True)
        else:
            df_dummified = pd.get_dummies(df, prefix=features_to_dummify, 
                                          columns=features_to_dummify, dummy_na=True)

        if col in categorical_features:
            aggregation_method = "mode"
        else:
            aggregation_method = "mean"

        print("Imputing column '{}' using the {} as aggregation method".format(col, aggregation_method))
        df[col] = knn_impute(target = df_dummified[col], attributes = df_dummified.drop(["inc", col], axis=1), 
                             aggregation_method = aggregation_method, k_neighbors = 10, numeric_distance = "euclidean", 
                             categorical_distance = "jaccard", missing_neighbors_threshold = 0.3)

        if df[col].isna().values.any():
            print("The column {} has not been successfully imputed as there are still missing values.".format(col))
        else:
            print("The column {} has been successfully imputed and does not contain missing values anymore.".format(col))
            
    return df

df = manual_knn_impute(df)

# %%  
df.info()

# %%  
df.to_csv("../data/df_breast_cancer.csv".format(date_notation), index=False)
