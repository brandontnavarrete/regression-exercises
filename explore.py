# imports to run my functions
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
# sql ace credentials
import env

# for chart image
from IPython.display import Image



def plot_variable_pairs(df):
    filename = 'pairplot10000.png'
    
    if os.path.isfile(filename):
        return Image(filename='pairplot10000.png')
    else:
        sns_plot = sns.pairplot(train.sample(10000), height = 2.0,corner = False, diag_kind = 'kde', kind = 'reg')
        sns_plot.savefig("pairplot10000.png")
        
        #Clean figure from sns 
        plt.clf()
        return Image(filename='pairplot.png') # Show pairplot as image