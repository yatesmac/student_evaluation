'''test.py - Test Flask application script.'''
import json
import logging
import sys

import requests


logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('../logs/predictions.log', mode='a'),
        logging.StreamHandler(sys.stdout)]
    )

url = 'http://localhost:9696/predict'
data = '../data/random_rows/row5778.json'

# Open the JSON file with sample row data 
with open(data) as f:
    sample_data = json.load(f)
print(sample_data)

response = requests.post(url, json=sample_data)
if response.status_code == 200:        
    probability = response.json()['probability']
    prediction = response.json()['prediction']
    logger.info(
        f'Probability: {probability:.3f} \nPrediction: {prediction} \n')
else:
    logger.info(
        f'Failed to retrieve prediction. Status Code: {response.status_code} \n')