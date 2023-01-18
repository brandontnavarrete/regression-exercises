# imports
import os
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly
import scipy.stats as stats
import statistics as s


# sklearn imports

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
#sse
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score


# sql ace credentials
import env
import wrangle as wr
import seaborn as sns
import explore as ex


def plot_residuals(y, yhat):
    ''' 
    takes in your data frame in the format as ([df['target'],df['yhat']) and creates residuals. returns a scatter plot
    '''
    residuals = y - yhat
    
    plt.scatter(x=y, y=residuals)
    plt.xlabel('Home Value')
    plt.ylabel('Residuals')
    plt.title('Residual vs Home Value Plot')
    plt.show()
    
# ||||||||||||||||||||||||||||||||||||||||||
def regression_errors(y, yhat):
    
    """ 
    takes in your data frame in the format as ([df['target'],df['yhat']) and creates residuals. returns a
    model mmse,sse,rmse,ess,tss

    """
    mse = mean_squared_error(y, yhat)
    sse=  mse * len(y)
    rmse = sqrt(mse)
    
    ess = ((yhat - y.mean())**2).sum()
    tss = ess + sse
    
    return sse, ess, tss, mse, rmse

# ||||||||||||||||||||||||||||||||||||||||||

def baseline_mean_errors(y):
    
    """
    takes (df['target'] and returns baseline sse, mse , rmse
    """
    
    baseline = np.repeat(y.mean(), len(y))
    
    mse = mean_squared_error(y, baseline)
    sse = mse * len(y)
    rmse = sqrt(mse)
    
    return sse, mse, rmse

# ||||||||||||||||||||||||||||||||||||||||||

def better_than_baseline(y, yhat):
    
    """ 
    calls function regression_erros and baseline_mean_errors. compares the model sse to the sse baseline and returns a true or 
    false statement
    """
    
    sse, ess, tss, mse, rmse = regression_errors(y, yhat)
    
    sse_baseline, mse_baseline, rmse_baseline = baseline_mean_errors(y)
    
    if sse < sse_baseline:
        print('My model performs better than baseline')
    else:
        print('My model performs worse than baseline')