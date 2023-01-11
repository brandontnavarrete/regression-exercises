# imports to run my functions
import os
import pandas as pd 
import numpy as np
# sql ace credentials
import env


# setting connectiong to sequel server using env

def get_connection(db, user=env.username, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


#----------------------------------------------
 
# acquiring zillow data using a get_connection

def get_zillow_data(get_connection):
    filename = "zillow.csv"
    
    # if the file already exist , return it. if not, then continue down the line.
    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col = 0)
    
    else:
        
    # read the SQL query into a dataframe
        df = pd.read_sql(
        '''
          select 
          bedroomcnt,
          bathroomcnt, calculatedfinishedsquarefeet,
          taxvaluedollarcnt,
          yearbuilt,
          taxamount,
          fips
          from properties_2017 
        ''', get_connection('zillow'))
        
        return df