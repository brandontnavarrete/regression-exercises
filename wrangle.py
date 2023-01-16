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




#----------------------------------------------

# setting connectiong to sequel server using env

def get_connection(db, user=env.username, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

#----------------------------------------------
 
# acquiring zillow data using a get_connection

def get_zillow_data():
    """returns a dataframe from SQL of all 2017 properties that are single family residential"""

    sql = """
    select 
    bedroomcnt, 
    bathroomcnt, 
    calculatedfinishedsquarefeet, 
    taxvaluedollarcnt, 
    yearbuilt,
    taxamount, 
    fips
    from properties_2017
    join propertylandusetype using (propertylandusetypeid)
    where propertylandusedesc = "Single Family Residential"
    """
    return pd.read_sql(sql, get_connection("zillow"))

#-------------------------------------------------

def change_zillow(df):
    
    df = df.dropna()
    
    df["fips"] = df["fips"].astype(int)
    
    df["yearbuilt"] = df["yearbuilt"].astype(int)
    
    df["bedroomcnt"] = df["bedroomcnt"].astype(int)
    
    df["taxvaluedollarcnt"] = df["taxvaluedollarcnt"].astype(int)
    
    df["calculatedfinishedsquarefeet"] = df["calculatedfinishedsquarefeet"].astype(int)
    
    return df

#-------------------------------------------------

def rename_cols(df):
    df = df.rename(columns={'bedroomcnt':'bedrooms', 
                            'bathroomcnt':'bathrooms', 
                            'calculatedfinishedsquarefeet':'sq_feet', 
                            'taxvaluedollarcnt':'tax_value',
                            'yearbuilt':'year_built',
                            'taxamount':'tax_amount'})
    return df

#-------------------------------------------------

def clean_zillow(df):
    
    '''
    takes data frame and changes datatypes and renames columnns, returns dataframe
    '''
    
    df = change_zillow(df)
    
    df = handle_outliers(df)
    
    df = rename_cols(df)

    df.to_csv("zillow.csv", index=False)

    return df

#-------------------------------------------------

def handle_outliers(df):
    '''handle outliers that do not represent properties likely for 99% of buyers and zillow visitors'''
    
    df = df[df.bathroomcnt <= 6]
    
    df = df[df.bedroomcnt <= 6]

    df = df[df.taxvaluedollarcnt < 2_000_000]

    df = df[df.calculatedfinishedsquarefeet < 10000]
    
    df = df[df.yearbuilt > 1850]

    return df

#-------------------------------------------------

def wrangle_zillow():
    """
   Acquires zillow data and uses the clean function to call other functions and returns a clean data        frame with new names, dropped nulls, new data types.
    """

    filename = "zillow.csv"

    if os.path.isfile(filename):
        df = pd.read_csv(filename)
    else:
        df = get_zillow_data()

        df = clean_zillow(df)

    return df

#-------------------------------------------------

def split_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames.
    return train, validate, test DataFrames.
    '''
    
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123)
    
    print(train.shape , validate.shape, test.shape)

          
    return train, validate, test


#-------------------------------------------------


def scale_data(train, 
               validate, 
               test, 
               values=['bedrooms', 'bathrooms', 'tax_amount', 'sq_feet'],
               return_scaler=False):
    '''
    Scales the 3 data splits. 
    Takes in train, validate, and test data splits and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    '''
    # make copies of our original data so we dont gronk up anything
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    #     make the thing
    scaler = MinMaxScaler()
    #     fit the thing
    scaler.fit(train[values])
    # applying the scaler:
    train_scaled[values] = pd.DataFrame(scaler.transform(train[values]),
                                                  columns=train[values].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[values] = pd.DataFrame(scaler.transform(validate[values]),
                                                  columns=validate[values].columns.values).set_index([validate.index.values])
    
    test_scaled[values] = pd.DataFrame(scaler.transform(test[values]),
                                                 columns=test[values].columns.values).set_index([test.index.values])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled

    

#-------------------------------------------------
 

