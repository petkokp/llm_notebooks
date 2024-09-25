import torch
import zipfile
from enum import Enum
import os
import subprocess
from .equation_generator import create_simple_arithmetic_dataset
from .tokenizer import get_encoding, encode, get_vocabulary

class DatasetNames(Enum):
    FRANKENSTEIN = "Frankenstein"
    ARITHMETIC = "Arithmetic"

class Dataset():
    def __init__(self):
        self.paths: dict[DatasetNames, str] = {
            DatasetNames.FRANKENSTEIN: "",
            DatasetNames.ARITHMETIC: "",
        }
        
    def generate(self, name: DatasetNames):
        if name == DatasetNames.FRANKENSTEIN:
            FILE_NAME = "4618-frankenshtajn.txt.zip"
            FRANKENSTEIN_URL = f"https://chitanka.info/text/{FILE_NAME}"
            
            subprocess.run(["wget", "-O", FILE_NAME, FRANKENSTEIN_URL])

            with zipfile.ZipFile(FILE_NAME, 'r') as zip_ref:
                zip_ref.extractall(".")
                
            BOOK_NAME = "Mary-Shelley -  - . Frankenshtajn - 4618.txt"
            DATASET_PATH = f"{name.value}_dataset.txt"
                
            os.rename(BOOK_NAME, DATASET_PATH)
            os.remove(FILE_NAME)
            
            self.paths[name] = DATASET_PATH
            return name               

        elif name == DatasetNames.ARITHMETIC:
            path = create_simple_arithmetic_dataset(100000)
            self.paths[DatasetNames.ARITHMETIC] = path
            return path
            
    def get(self, name: DatasetNames, split_train_coefficient: 0.9):
        dataset_path = self.paths[name]
        
        with open(dataset_path) as file:
            text = file.read()
            
        vocabulary = get_vocabulary(text)
        encoding = get_encoding(vocabulary)
        
        data = torch.tensor(encode(text, encoding))

        train_data_size = round(len(data) * split_train_coefficient)

        train_data = data[:train_data_size]
        
        val_data = data[train_data_size:]

        return train_data, val_data