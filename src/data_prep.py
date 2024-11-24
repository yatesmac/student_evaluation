'''dataprep.py - Prepare data for training: Create training, validation and testing datasets.'''
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer


def drop_col(col_names_list, df): 
    ''' Drop multiple columns based on their column names.'''
    df.drop(columns=col_names_list, axis=1, inplace=True)
    return df


def check_missing_data(df):
    '''Remove any missing data in the df.'''
    if df.isna().sum().sum():
        df = df.dropna()
    return df


def format_text(df):
    '''Format all text values - removing spaces and hyphens.'''
    for c in categorical:
        df[c] = df[c].str.lower().str.replace(' ', '_').replace('-', '_', regex=True)
    return df


def format_numbers(df):
    '''Format all numerical values.'''
    for n in numerical:
        df[n] = pd.to_numeric(df[n], errors='coerce')
    return df


def split_data(df, y):
    '''Create Training, Validation and Testing datasets.''' 
    df_train, df_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=1, stratify=y)
    dv = DictVectorizer(sparse=False)

    train_dict = df_train[categorical + numerical].to_dict(orient='records')
    dv = dv.fit(train_dict)
    X_train = dv.transform(train_dict)

    test_dict = df_test[categorical + numerical].to_dict(orient='records')
    X_test = dv.transform(test_dict)

    return dv, X_train, X_test, y_train, y_test


def prepare_data(df):
    '''Clean data before splitting it.'''
    target = 'exam_score' # Target Variable
    high_corr = 'attendance' # Variable that has a high correlation to our target
    
    df = check_missing_data(df)
    # Remove invalid rows.
    df = df[df.exam_score <= 100]
    y = np.where(df[target] <= 65, 0, 1)
                 
    # Remove Target and Highly correlated variable from dataset
    cols = [target, high_corr]
    df = drop_col(cols, df)
    for c in cols:
        numerical.remove(c)

    df = format_text(df)
    df = format_numbers(df)
    return df, y


# Data Preparation
data = '../data/StudentPerformanceFactors.csv'
X = pd.read_csv(data)

X.columns = X.columns.str.lower().str.replace(' ', '_') # All columns names
categorical = list(X.dtypes[X.dtypes == 'object'].index) # Categorical columns
numerical = list(X.dtypes[X.dtypes != 'object'].index) # Numeric columns

X, y = prepare_data(X)
dv, X_train, X_test, y_train, y_test = split_data(X, y)

