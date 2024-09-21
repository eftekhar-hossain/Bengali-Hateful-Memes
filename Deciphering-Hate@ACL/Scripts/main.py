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
    if args.task == 'task1':
      print(classification_report(true,pred,target_names=['not-hate','hate'],digits = 3))
    else:
      print(classification_report(true,pred,target_names=['TI','TC','TO','TS'],digits = 3))

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
    # print(saved_models_dir)
    

    ## Load the Processed Data Splits
    train_loader, valid_loader, test_loader, class_weights = d.load_dataset(excel_path,
                                                              memes_path, 
                                                              args.task, 
                                                              args.maximum_length,
                                                              args.batch)
                                                        
    #train the model 
    start_time = time.time()
    actual, pred = t.pipline(train_loader, 
                              valid_loader, 
                              test_loader, 
                              args.task,
                              saved_models_dir,
                              args.n_heads,
                              class_weights,
                              args.epochs, 
                              args.lr_rate)

    # #  evaluation
    print(f"Classification Report for {args.task}:")
    print_metrices(actual, pred)

    end_time = time.time()
    print(f"Total time :{end_time-start_time:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bengali Hateful Memes Classification')

    parser.add_argument('--task', dest='task', type=str, default = 'task1',
                        help='Binary (task1) or Multiclass (task2) Classification')
    parser.add_argument('--dataset', dest='dataset_path', type=str, default = 'BHM',
                        help='the directory of the dataset folder')
    parser.add_argument('--max_len', dest='maximum_length', type=int, default = 50,
                        help='the maximum text length')
    parser.add_argument('--batch_size',dest="batch", type=int, default = 4,
                        help='Batch Size - default 4')   
    parser.add_argument('--model', dest='model_path', type=str, default = 'Saved_Models',
                        help='the directory of the saved model folder')
    parser.add_argument('--heads',dest="n_heads", type=int, default = 2,
                        help='number of heads - default 2')                       
    parser.add_argument('--n_iter',dest="epochs", type=int, default = 1,
                        help='Number of Epochs - default 1')
    parser.add_argument('--lrate',dest="lr_rate", type=float, default = 2e-5,
                        help='Learning rate - default 2e-5')
                     

    
    args = parser.parse_args()
    main(args)