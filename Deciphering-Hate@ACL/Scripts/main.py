import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re,nltk,json
from bs4 import BeautifulSoup
### ML Librarires--------------------
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from sklearn.metrics import average_precision_score,roc_auc_score, roc_curve, precision_recall_curve
###-------------------------------------------
np.random.seed(42)
import random
import warnings
warnings.filterwarnings('ignore')

import sys
import argparse
import time
import dataset as d
import architecture as m
import training as t


'''Evaluation Parameters'''

def print_metrices(true,pred):
    # print(confusion_matrix(true,pred))
    print(classification_report(true,pred,target_names=['not-hate','hate'],digits = 3))
    # print("Accuracy : ",accuracy_score(true,pred))
    # print("Precison : ",precision_score(true,pred, average = 'weighted'))
    # print("Recall : ",recall_score(true,pred,  average = 'weighted'))
    # print("F1 : ",f1_score(true,pred,  average = 'weighted'))
    # print("ROC-AUC: ",roc_auc_score(true, pred))



def main(args):
    start_time = time.time()
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming the root directory is one levels up from where the script is located
    root_dir = os.path.abspath(os.path.join(script_dir, ".."))
    # Construct the path to the dataset folder relative to the script directory
    dataset_base_path = os.path.join(root_dir, args.dataset_path)
    # Construct the full path by appending the 'Files' folder and the Excel file
    excel_path = os.path.join(dataset_base_path, 'Files')
    memes_path = os.path.join(dataset_base_path, 'Memes')
    # create a path model saving
    saved_models_dir = os.path.join(root_dir, args.model_path)
    # Create the folder if it doesn't already exist
    os.makedirs(saved_models_dir, exist_ok=True)
    # print(excel_path)
    

    # Load the Processed Data Splits
    train_loader, valid_loader, test_loader = d.load_dataset(excel_path,memes_path)
    # train the model 

    actual, pred = t.pipline(train_loader, valid_loader, test_loader, args.epochs, args.lr_rate)

    #  evaluation
    print("Classification Report:")
    print_metrices(actual, pred)

    end_time = time.time()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bengali Hateful Memes Classification')

    parser.add_argument('--dataset', dest='dataset_path', type=str, default = 'BHM',
                        help='the directory of the dataset folder')
    parser.add_argument('--model', dest='model_path', type=str, default = 'Saved_Models',
                        help='the directory of the dataset folder')                    

    parser.add_argument('--n_iter',dest="epochs", type=int, default = 1,
                        help='Number of Epochs - default 1')

    parser.add_argument('--lrate',dest="lr_rate", type=float, default = 2e-5,
                        help='Learning rate - default 2e-5')

   # length of the text
   # classification task number
   # number of heads                     
   # batch size 
    
    args = parser.parse_args()
    main(args)