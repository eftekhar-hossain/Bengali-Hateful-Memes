import pandas as pd
import numpy as np
import os
import pickle as pkl
from tqdm import tqdm
from PIL import Image, ImageFile
import tensorflow as tf
import tensorflow.keras as keras
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



  ## collect image names from the folders
def create_img_array(img_dirct):
    all_imgs = []
    for root, j, files in os.walk(img_dirct):
        for file in files:
            file = root + '' + file
            all_imgs.append(file)
    return all_imgs

def create_img_path(DF, Col_name, img_dir):
    img_path = [img_dir + '' + str(name) for name in DF[Col_name]]
    return img_path

# Function that returns image reading from the path
def get_input(path,size):
    # Loading image from given path
    # and resizing it to desired format
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img = tf.keras.utils.load_img(path, target_size=(size,size))
    return(img)

# Takes in image and preprocess it
def process_input(img):
    # Converting image to array
    img_data = tf.keras.utils.img_to_array(img)
    # Adding one more dimension to array
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    return(img_data)

def image_pickle(img_path, size, dataset_name, split_name):
# Create an array of training images
    images = []
    for i in tqdm(range(len(img_path)),desc=f'Making pickle {split_name} images'):
        input_img = get_input(img_path[i],size)
        input_img = process_input(input_img)
        images.append(input_img[0])

    # convert into numpy array
    image = np.array(images)
    with open('Datasets/'+ f'{split_name}_image_{dataset_name}.pkl','wb') as f:
      pkl.dump(image, f)
      print('Saved.')  


def TextTokenizer(train, valid, test, max_len, txt_column_name, dataset_name, split_name):

  tokenizer = Tokenizer(num_words = 50000 ,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n-',
                        split=' ', char_level=False, oov_token='<oov>', document_count=0)
  tokenizer.fit_on_texts(train[txt_column_name])
  word_index = tokenizer.word_index
  print("Vocab Size: ",len(word_index)+1)

  # Training Sequences
  train_sequences = tokenizer.texts_to_sequences(train[txt_column_name])
  train_pad_sequences =  pad_sequences(train_sequences, value=0.0, padding='post', maxlen= max_len)

  # Validation Sequences
  valid_sequences = tokenizer.texts_to_sequences(valid[txt_column_name])
  valid_pad_sequences =  pad_sequences(valid_sequences, value=0.0, padding='post', maxlen= max_len)

  # Test Sequences
  test_sequences = tokenizer.texts_to_sequences(test[txt_column_name])
  test_pad_sequences =  pad_sequences(test_sequences, value=0.0, padding='post', maxlen= max_len)

  seq = [train_pad_sequences, valid_pad_sequences, test_pad_sequences]

  for i in tqdm(range(len(seq)),desc = "Making pickles for text data: "):
    with open('Datasets/'+ f'{split_name[i]}_text_{dataset_name}.pkl','wb') as f:
        pkl.dump(seq[i], f)
  print('Saved.')   



def ImgToArray(img_dir, train_data, valid_data, test_data, image_column_name, img_size, dataset_name, split_name):
  
  train_img_path = create_img_path(train_data,image_column_name, img_dir)
  valid_img_path = create_img_path(valid_data,image_column_name, img_dir)
  test_img_path = create_img_path(test_data,image_column_name, img_dir) 
  size = img_size
  dataset_name = dataset_name
  split_name = split_name
  
  image_pickle(train_img_path, size, dataset_name, split_name[0])
  image_pickle(valid_img_path, size, dataset_name, split_name[1])
  image_pickle(test_img_path, size, dataset_name, split_name[2])







