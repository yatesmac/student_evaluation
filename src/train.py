'''train.py - Train Logistic Regression model and save to Pickle file.'''
import pickle
import logging
import sys

import os.path

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score

import data_prep as dp


logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('../logs/debug.log', mode='a'),
        logging.StreamHandler(sys.stdout)]
    ) 

model_file = '../models/dv_model.pkl'
params = {
    'max_iter': 200, 
#   'C': 1, 
    'cv': 3,
    'penalty': 'l2',
    'random_state': 1}


def train(params):
    '''Train the final model.'''
    logger.info(
        'Training the final model using Logistic Regression with cross validation.')
    model = LogisticRegressionCV(**params)
    return model.fit(dp.X_train, dp.y_train)

    
def model_score(model):
    '''Validate model prfomance.'''   
    y_pred = model.predict_proba(dp.X_test)[:, 1]
    roc_score = roc_auc_score(dp.y_test, y_pred)
    logger.info(f'ROC AUC on test dataset = {roc_score} \n')
    return roc_score


def save_model(model):
    '''Save the model.'''
    logger.info('Saving the model and data vectorizer as a pickle file')
    try: 
        with open(model_file, 'wb') as f:
            pickle.dump((dp.dv, model), f)
        logger.info(f'Model saved successfully to {model_file} \n')
    
    except OSError as e:
        print(e, '\n')
    

def main():
    '''Train, evaluate and save model.'''
    model = train(params)

    roc_score = model_score(model)
    print(f'ROC_AUC = {roc_score}')
    
    if not os.path.exists(model_file):
        save_model(model)


if __name__ == '__main__':
    main()