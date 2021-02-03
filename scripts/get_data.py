# -*- coding: utf-8 -*-
"""
get_data.py
Functions for reading data.

Created on Thu Mar 19 12:34:34 2020
"""
# Import packages.
import pathlib
import pandas as pd


#%%
def survival(path_data):
    """
    Get the relevant data.

    Parameters
    ----------
    path_data: string or Pathlib path
        Path to the data directory.

    Returns
    -------
    df: Data frame
    """
    
    if isinstance(path_data, str):
        path_data = pathlib.Path(path_data)
        
    df = pd.read_csv(path_data/'df_breast_cancer.csv')

    return df