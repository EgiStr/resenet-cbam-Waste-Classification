import os
import kaggle

def download_dataset():
    api = kaggle.KaggleApi()
    api.authenticate()
    api.dataset_download_files('techsash/waste-classification-data', path='data', unzip=True)
    print("Dataset downloaded and extracted to data/")

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    download_dataset()