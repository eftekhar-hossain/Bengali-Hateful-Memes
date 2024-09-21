import pandas as pd
import numpy as np
import os
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoModel,AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import clip

# Set the device to GPU if available, else use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:",device)

start_time = time.time()

def balanced_weigths(data):
      # class wights
      class_weights = compute_class_weight('balanced', classes=np.unique(data['Labels']), y=data['Labels'])
      class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
      return class_weights


def load_dataset(files_path,memes_path, task, max_len, batch_size):
    # Paths
    
    if task == 'task1':
      print("Fetching Binary Dataset... ")
      print("Maximum Text Length: ", max_len)
      print("Batch Size: ", batch_size)
      train_file = os.path.join(files_path, "train_task1.xlsx")
      valid_file = os.path.join(files_path, "valid_task1.xlsx")
      test_file = os.path.join(files_path, "test_task1.xlsx")
    
      # binary dataset
      train_data = pd.read_excel(train_file)
      valid_data = pd.read_excel(valid_file)
      test_data = pd.read_excel(test_file)

      # encode labels

      train_data['Labels'] = train_data['Labels'].replace({"non-hate":0,"hate":1})
      valid_data['Labels'] = valid_data['Labels'].replace({"non-hate":0,"hate":1})
      test_data['Labels'] = test_data['Labels'].replace({"non-hate":0,"hate":1})

      weights = balanced_weigths(train_data)

    else:
      print("Fetching Multiclass Dataset... ")
      print("Maximum Text Length: ", max_len)
      print("Batch Size: ", batch_size)
      train_file = os.path.join(files_path, "train_task2.xlsx")
      valid_file = os.path.join(files_path, "valid_task2.xlsx")
      test_file = os.path.join(files_path, "test_task2.xlsx")
    
      # binary dataset
      train_data = pd.read_excel(train_file)
      valid_data = pd.read_excel(valid_file)
      test_data = pd.read_excel(test_file)

      # encode labels

      train_data['Labels'] = train_data['Labels'].replace({"TI":0,"TC":1,"TO":2,"TS":3})
      valid_data['Labels'] = valid_data['Labels'].replace({"TI":0,"TC":1,"TO":2,"TS":3})
      test_data['Labels'] = test_data['Labels'].replace({"TI":0,"TC":1,"TO":2,"TS":3})
      weights = balanced_weigths(train_data)
      

    
    # print("Training Data:", len(train_data))
    # print("Valid Data:", len(valid_data))
    # print("Test Data", len(test_data))



        # Load your dataset (Assuming you have a CSV file)
    class BHMDataset(Dataset):
        def __init__(self, dataframe, tokenizer, data_dir, max_seq_length, transform=None):
            self.data = dataframe
            self.max_seq_length = max_seq_length
            self.data_dir = data_dir
            self.tokenizer = tokenizer
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            img_name = os.path.join(self.data_dir, self.data.loc[idx, 'image_name'])
            image = Image.open(img_name)
            caption = self.data.loc[idx, 'Captions']
            label = int(self.data.loc[idx, 'Labels'])

            if self.transform:
                image = self.transform(image)

            # Tokenize the caption using BERT tokenizer
            inputs = self.tokenizer(caption, return_tensors='pt',
                                    padding='max_length', truncation=True, max_length=self.max_seq_length)

            return {
                'image': image,
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
                'label': label
            }

    # Data preprocessing and augmentation
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Initialize BERT
    tokenizer = AutoTokenizer.from_pretrained("facebook/xglm-564M")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    # Convert model weights to the same data type as the input data
    clip_model = clip_model.half()

    print("Preparing Data Loaders...")
    # Create data loaders
    train_dataset = BHMDataset(dataframe = train_data, tokenizer = tokenizer,data_dir = memes_path,
                                max_seq_length=max_len,transform=data_transform )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = BHMDataset(dataframe = valid_data, tokenizer = tokenizer,data_dir = memes_path,
                                max_seq_length=max_len,transform=data_transform )
    val_loader = DataLoader(val_dataset, batch_size=batch_size,shuffle=False)

    test_dataset = BHMDataset(dataframe = test_data, tokenizer = tokenizer,data_dir = memes_path,max_seq_length=max_len,transform=data_transform )
    test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False)

    print("Done.")
    end_time = time.time()

    print(f"Time required for preparing the Data loaders: {end_time-start_time:.2f}s")

    # check a loader

    # for batch in val_loader:
    #   images = batch['image'].to(device)
    #   input_ids = batch['input_ids'].to(device)
    #   attention_mask = batch['attention_mask'].to(device)
    #   labels = batch['label'].to(device)

    #   print("Images: ",images.shape,"\n","Input_IDs: ",input_ids.shape,"\n","Attention Mask: ",attention_mask.shape,"\n","Labels: ",labels.shape)
    # # Now you can inspect the shapes for this single example
    #   break  # Exit the loop after the first batch (1 example)
    return train_loader,val_loader,test_loader, weights


