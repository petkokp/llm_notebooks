import zipfile
from enum import Enum
import os
import subprocess
from .equation_generator import create_simple_arithmetic_dataset

class DatasetNames(Enum):
    FRANKENSTEIN = "Frankenstein"
    ARITHMETIC = "Arithmetic"

class Dataset():
    def __init__(self):
        self.paths: dict[DatasetNames, str] = {
            DatasetNames.FRANKENSTEIN: "",
            DatasetNames.ARITHMETIC: "",
        }
        
    def get(self, name: DatasetNames):
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
            
    
