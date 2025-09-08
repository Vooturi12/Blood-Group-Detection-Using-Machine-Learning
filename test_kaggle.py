# test_kaggle.py
import os
import kaggle.api
from kaggle.api.kaggle_api_extended import KaggleApi

# Set Kaggle credentials
os.environ['KAGGLE_USERNAME'] = 'abhayankith'
os.environ['KAGGLE_KEY'] = '0afdc9d323831c2dc6db7bbe973d0cb1'

# Important: We need to skip the initial automatic authentication in kaggle.__init__
# And manually initialize and authenticate the API
api = KaggleApi()
api.authenticate()

# Test authentication
try:
    print("Authentication successful!")
    
    # List some datasets to verify it works fully
    datasets = api.dataset_list(search="fingerprint")
    print(f"Found {len(datasets)} datasets")
    
    for dataset in datasets[:3]:
        print(f"- {dataset.ref}")
    
except Exception as e:
    print(f"Error: {e}")