# How to Run

The following instructions will show how to run the model with the **MUTE** dataset. 

1. Clone this repository and navigate to Align-Before-Attend@EACL folder
```bash
git clone https://github.com/eftekhar-hossain/Bengali-Hateful-Memes.git
cd Bengali-Hateful-Memes
cd Align-Before-Attend@EACL
```

2. Install Packages
```Shell
pip install -r requirements.txt  # download the required libraries
```
3. Download the MUTE Dataset
   
```python
import gdown
# Replace 'YOUR_FILE_ID' with the actual file ID from the Google Drive link.
gdown.download("https://drive.google.com/uc?export=download&id=1ozTFUM7q27g7uckhPWUiQFwhROCiEUAc", "file.zip", quiet=False)
```
After running the cell, the dataset will be downloaded as **file.zip**

4. Unzip the file.zip

```python
import zipfile
zip_ref = zipfile.ZipFile("file.zip", 'r')
zip_ref.extractall()
zip_ref.close()
```
  After unzipping the file, you will see the **MUTE** dataset in the current directory. In the **MUTE** folder, you can see   three Excel files and one meme folder. To make the work simple, we will make pickle files of the visual and textual information for each dataset split.



5. Create a Datasets directory to save the pickle files 
```python
import os
os.mkdir("Datasets/")
```

After completing the above steps, your folders should organize the data as follows in `Bengali-Hateful-Memes/Align-Before-Attend@EACL/`,

```
├── Code
│   └── .py files
├── Datasets
│   └── will store .pkl files
├── MUTE
│   └── Memes
│         ├── .jpg
          └── .png
    ├── test_hate.xlsx
    └── train_hate.xlsx
    └── valid_hate.xlsx 
   
```

### Make Pickle files
Run the following scripts to create the pickle files for all the splits. The pickle files will be saved in the `./Datasets/` folder with appropriate names.

1. Generate Pickle files for train, validation, and test set images

```Shell
python Code/main.py -d-img \
-img_dir 'MUTE/Memes/' \
-train 'MUTE/train_hate' \
-valid 'MUTE/valid_hate' \
-test 'MUTE/test_hate' \
-img_column_name 'image_name'  \
-img_size 150 \
-dataset_name 'mute' \
-split_name ['train','valid','test']
```
Arguments:

- `-d-img`: read the images from the datasets.
- `-img_dir`: provide the image directory path as a string.
- `-train`: train set Excel file name with directory. (exclude the `.xlsx`) 
- `-valid`: validation set Excel file name with directory. (exclude the `.xlsx`)
- `-test`: test set Excel file name with directory. (exclude the `.xlsx`)
- `-img_column_name`: give the column name where the image names are stored.
- `-img_size`: desired size of the input image.
- `-dataset_name`: name of the dataset.
- `-split_name`: pass the list of split names

2. Generate Pickle files for training, validation, and test set captions. You will also obtain the `vocabulary size` that can be used during model training.

```Shell
python Code/main.py -d-txt \
-train 'MUTE/train_hate' \
-valid 'MUTE/valid_hate' \
-test 'MUTE/test_hate' \
-txt_len 60 \
-txt_column_name 'Captions'  \
-dataset_name 'mute' \
-split_name ['train','valid','test']
```
Arguments:

- `-d-txt`: read the texts from the datasets.
- `-txt_len`: maximum text length for padding.
- `-txt_column_name`: the column name where the text is stored 


### Models Training and Evaluation
You can run and evaluate the proposed method and its variants on the MUTE dataset.

```Shell
python Code/main.py -models \
-method_name 'mca-scf' \
-datasets ['MUTE/train_hate','MUTE/valid_hate','MUTE/test_hate'] \
-train_pkl ['train_image_mute','train_text_mute'] \
-valid_pkl ['valid_image_mute','valid_text_mute'] \
-test_pkl ['test_image_mute','test_text_mute'] \
-label_column 'Label'  \
-hparams "[60, 150, 11993, 32, 2]" \
-md_hparams "['mca-scf_mute', 32, 3]" \
```
Arguments:

- `-models`: represent the models' training.
- `-method_name`: Provide the name of the method you want to build. The paper described one proposed method and three variants of the proposed method. The method names you can pass **`mca-scf`**, **`mcf`**, **`tgcf`**, and **`vgcf`**. Please read the paper to get the full information about these methods.
- `-datasets`: list of Excel file names of the training, validation, and test set. (Exclude `.xlsx` from the name)
- `-train_pkl`: list of saved train pickle file names. It should be passed in the order of `image pickles` and `text pickles`.
- `—valid_pkl`: A list of saved valid pickle file names. It should be passed in the order of `image pickles` and `text pickles`.
- `-test_pkl`: list of saved test pickle file names. It should be passed in the order of `image pickles` and `text pickles`.
- `-label_column`: Column name where the class labels are stored in the Excel files.
- `-hparams`: list of hyperparameters. The list represents `[maximum text length, image size, vocabulary size, embedding dimension, number of classes]`.
- `-md_hparams`: list of model hyperparameters. The list represents `[model name to saved, batch size, number of epochs]`.


**Demo code** [Colab Notebook](https://github.com/eftekhar-hossain/Bengali-Hateful-Memes/blob/main/Align-Before-Attend@EACL/demo_code_%5BEACL_SRW'24%5D.ipynb) 
