# imports to run my functions
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
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
    
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||    
    
def plot_categorical_and_continuous_vars(df):
      
    # making local variables
    cont_var = ['sq_feet','tax_value','tax_amount']

    disc_var = ['bedrooms','bathrooms','year_built','fips']

    # for loop for continous variable histograms
    for col in cont_var:
        sns.displot(df[col],bins = 100)
        plt.title(f'{col} continous variable distribution')
   

        plt.ylim(0, 40)
    
        plt.show()
    
    for col in df.columns:
        plt.figure(figsize = (16,6))
        sns.boxplot(x = df[col],data=df)
        plt.title (f'{col} outliers')
    
        plt.show()
    
    for col in disc_var:
        sns.displot(df[col],bins = 100)
        plt.title(f'{col} discrete variable distribution')
   

        plt.ylim(0, 40)
    
        plt.show()
        
        
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||    

        
def heat_map(train):
    
    train_corr = train.corr('spearman')
    train_corr
    
    sns.heatmap(train_corr, cmap='mako', annot=True, 
            mask=np.triu(train_corr))
