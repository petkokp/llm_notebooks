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
    DOSTOEVSKY = "Dostoevsky"

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
        elif name == DatasetNames.DOSTOEVSKY:
            urls = ["https://chitanka.info/text/15682-bratja-karamazovi.txt.zip", "https://chitanka.info/book/3180-idiot.txt.zip", "https://chitanka.info/book/1445-prestyplenie-i-nakazanie.txt.zip", "https://chitanka.info/book/4171-besove.txt.zip", "https://chitanka.info/book/6578-beli-noshti.txt.zip"]
        
            for url in urls:
                file_name = url.split("/")[-1]
                subprocess.run(["wget", "-O", file_name, url])
                
                with zipfile.ZipFile(file_name, 'r') as zip_ref:
                    zip_ref.extractall(".")
                    
            output_file = "dostoevsky_dataset.txt"
            
            txt_files = [file for file in os.listdir() if file.endswith(".txt")]

            with open(output_file, "w") as outfile:
                for file in txt_files:
                    with open(file, "r") as infile:
                        outfile.write(infile.read())
                        outfile.write("\n")
                        
            self.paths[name] = output_file
            return output_file
        
        elif name == DatasetNames.ARITHMETIC:
            path = create_simple_arithmetic_dataset(10000000)
            self.paths[DatasetNames.ARITHMETIC] = path
            return path
            
    def get(self, name: DatasetNames, split_train_coefficient=0.9):
        dataset_path = self.paths[name]
        
        with open(dataset_path) as file:
            text = file.read()
            
        vocabulary = get_vocabulary(text)
        encoding = get_encoding(vocabulary)
        
        data = torch.tensor(encode(text, encoding))

        train_data_size = round(len(data) * split_train_coefficient)

        train_data = data[:train_data_size]
        
        val_data = data[train_data_size:]
        
        vocabulary_size = len(vocabulary)

        return train_data, val_data, vocabulary_size