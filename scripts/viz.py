# -*- coding: utf-8 -*-
"""
viz.py
Functions for data visualization (i.e., plotting)

Created on Wed Jun 10 12:09:09 2020
"""
import os
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import random
from joblib import load
from statannot import add_stat_annotation

import shap
shap.initjs()

path_tools = os.path.join('C:\\', 'Users', 'XXXXX', 'Documents', 'IKNL', 'Projects', 'Tools')
if path_tools not in sys.path:
    sys.path.append(path_tools)
import helpers


#%%
def plot_c_index(c_index, fig=None, ax=None):
    """
    Generate plot of c-index results.

    Parameters
    ----------
    c_index: dictionary
        Each item corresponds to a different method.
        
    fig: figure handle
    
    ax: axes handle

    Returns
    -------
    fig, ax
        Handles to figure and axes of the plot.    
    """  
    
    df_c = pd.DataFrame(c_index)    
    df_c = pd.melt(df_c, value_vars=df_c.columns, var_name='method', value_name='c_index')
    
    # Create figure (if necessary).
    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 1, figsize=[7.5, 5])
    elif fig is None:
        fig = ax.get_figure()
    elif ax is None:
        ax = fig.gca()
    
    column_order = ['cph', 'rsf', 'ssvm', 'xgb']
    ax = sns.barplot(x='method', y='c_index', data=df_c, 
                     order=column_order,
                     ci='sd', 
                     capsize=0.02, 
                     palette='Blues', 
                     ax=ax)
    ax.set_xticklabels([x.upper() for x in column_order])
    ax, test_results = add_stat_annotation(ax, data=df_c, x='method', y='c_index', 
                                           order=column_order, 
                                           box_pairs=[('cph', 'xgb'), ('rsf', 'xgb'), ('ssvm', 'xgb')],
                                           test='t-test_paired', text_format='star', 
                                           comparisons_correction='bonferroni', 
                                           loc='outside', verbose=2)    
    
    ax.set_ylim([0, 0.8])
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
    ax.set_xlabel("Method")
    ax.set_ylabel("$c$-index")
        
    return fig, ax



#%%
def plot_shap(shap_values, X_test, fig=None, ax=None):
    """
    Generate plot of SHAP values.

    Parameters
    ----------
    shap_values:
        SHAP values
        
    X_test: pandas DataFrame
        Test data. Must correspond to the data used to compute the 
        SHAP values.
        
    fig: figure handle (optional)
    
    ax: axes handle (optional)

    Returns
    -------
    fig, ax
        Handles to figure and axes of the plot.    
    """  
    
    # Create figure (if necessary).
    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 1, figsize=[7, 5])
    elif fig is None:
        fig = ax.get_figure()
    elif ax is None:
        ax = fig.gca()
        
    shap.summary_plot(shap_values, X_test, show=False)
    ax.set_xlabel("SHAP value")
    ax.set_xlim([-2.25, 2.25]) # Adapt limits to results.
    plt.show()
    
    return fig, ax


#%%
def plot_shap_interactions(shap_values_interactions, X_test):
    """
    Generate plot of SHAP interaction values.

    Parameters
    ----------
    shap_values:
        SHAP values
        
    X_test: pandas DataFrame
        Test data. Must correspond to the data used to compute the 
        SHAP values.

    Returns
    -------
    fig, ax
        Handles to figure and axes of the plot.    
    """  
        
    # Make sure all variables will be plotted.
    n_features = len(X_test.columns)
    shap.summary_plot(shap_values_interactions, X_test, max_display=n_features, show=False)
    
    fig = plt.gcf()
    ax = plt.gcf()
    
    plt.show()
    
    return fig, ax


#%%
def plot_shap_dependence(col, shap_values, X_test, fig=None, ax=None):
    """
    Generate plot of SHAP dependence values.

    Parameters
    ----------
    col: string
        Column name of the variable to analyze.
        
    shap_values: list
        Each element has an array of SHAP values.
        Each group of SHAP values will be plotted with a different color.
        
    X_test: pandas DataFrame
        Test data. Must correspond to the data used to compute the 
        SHAP values.
        
    fig: figure handle (optional)
    
    ax: axes handle (optional)

    Returns
    -------
    fig, ax
        Handles to figure and axes of the plot.    
    """  
    
    # Create figure (if necessary).
    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 1, figsize=[2.5, 2.5])
    elif fig is None:
        fig = ax.get_figure()
    elif ax is None:
        ax = fig.gca()
        
    X_test_decoded = helpers.decode(X_test)
    
    # Get right colors.
    interval = np.linspace(0, 1, len(shap_values)+1)
    colors = [mpl.cm.Blues(x) for x in interval][1:]
    
    # Flip list order to make sure they are plotted in the right order
    # (XGB in the back with darker color, CPH in the front with lighter color)
    shap_values.reverse()
    colors.reverse()

    plt.axhline(y=0, xmin=-10, xmax=10, linewidth=2.5, linestyle='--', color=[0.6, 0.6, 0.6])
    for (ii, shap_values_curr), color in zip(enumerate(shap_values), colors):        
        shap.dependence_plot(col, shap_values_curr + (0.1*ii), X_test, display_features=X_test_decoded, 
                             color=color,
                             interaction_index=None,
                             alpha=0.5,
                             dot_size=7.5,
                             x_jitter=1,
                             ax=ax, show=False)
    ax.set_xlabel(" ")
    ax.set_ylabel(" ")
    ax.set_ylim([-1.5, 2.5])


    # Specific formatting per feature.
    if col=='age':
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(20))
    elif col=='pts':
        pass
    elif col=='ptmm':
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(50))
    elif col=='grd':
        ax.set_ylabel("SHAP value")
    elif col=='ply':
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(10))
    elif col=='horm':
        pass
    elif col=='ratly':
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.25))
    elif col=='rly':
        pass
    elif col=='mor':
        cph_line = mpl.lines.Line2D([], [], color=colors[1], marker='o',
                                  label='CPH')
        xgb_line = mpl.lines.Line2D([], [], color=colors[0], marker='o',
                                  label='XGB')
        plt.legend(handles=[cph_line, xgb_line], markerscale=1, frameon=False, fontsize='xx-small')

    plt.show()
    
    
    return fig, ax   
    

#%%
def plot_shap_dependence_int(col, shap_values, X_test, fig=None, ax=None):
    """
    Generate plot of SHAP dependence values.

    Parameters
    ----------
    col: string
        Column name of the variable to analyze.
        
    shap_values:
        SHAP values
        
    X_test: pandas DataFrame
        Test data. Must correspond to the data used to compute the 
        SHAP values.
        
    fig: figure handle (optional)
    
    ax: axes handle (optional)

    Returns
    -------
    fig, ax
        Handles to figure and axes of the plot.    
    """  
    
    # Create figure (if necessary).
    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 1, figsize=[3, 3])
    elif fig is None:
        fig = ax.get_figure()
    elif ax is None:
        ax = fig.gca()
        
    X_test_decoded = helpers.decode(X_test)

    plt.axhline(y=0, xmin=-10, xmax=10, linewidth=2.5, linestyle='--', color=[0.6, 0.6, 0.6])
    shap.dependence_plot(col, shap_values, X_test, display_features=X_test_decoded, 
                         alpha=0.5,
                         dot_size=5,
                         x_jitter=1,
                         ax=ax, show=False)
    ax.set_xlabel(" ")
    ax.set_ylabel(" ")
    
    # Specific formatting per feature.
    if col=='age':
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(20))
    elif col=='pts':
        pass
    elif col=='ptmm':
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(50))
    elif col=='grd':
        ax.set_ylabel("SHAP value")
    elif col=='ply':
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(10))
    elif col=='horm':
        pass
    elif col=='ratly':
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.25))
    elif col=='rly':
        pass
    elif col=='mor':
        pass

    plt.show()
    
    
    return fig, ax