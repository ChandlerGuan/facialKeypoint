# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 16:20:11 2017

@author: chandler
"""

import os
import numpy as np
from sklearn.utils import shuffle
from pandas.io.parsers import read_csv
import h5py;

FTRAIN = 'training.csv'
FTEST = 'test.csv'

h5_name = "kaggle.h5"

def load(test=False, cols=None):
    '''Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    '''
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    # print(df.count())  # prints the number of values for each column
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y


def load2d(test=False, cols=None):
    X, y = load(test=test, cols=cols)
    X = X.reshape(-1, 1, 96, 96)

    return X, y
    
def create_h5():
    x,y = load2d();
    h5_file = h5py.File(h5_name);
    h5_file['image'] = x;
    h5_file['landmark'] = y;
    h5_file.close();
    
    listFile = open('kaggle.txt','w');
    listFile.write(h5_name+'\n');
    listFile.close();
    
    
if __name__ == "__main__":
    create_h5();