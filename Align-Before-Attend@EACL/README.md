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
After running the cell the dataset will be downloaded as **file.zip**

4. Unzip the file.zip

```python
import zipfile
zip_ref = zipfile.ZipFile("file.zip", 'r')
zip_ref.extractall()
zip_ref.close()
```
  After unzipping the file you can see the **MUTE** dataset in the current directory. In the **MUTE** folder you can see   three Excel files and one folder consisting of memes. To make the work simple we will make pickle files of the visual and textual information for each dataset split.



5. Create a Datasets directory to save the pickle files 
```python
import os
os.mkdir("Datasets/")
```

After completing the above steps, your folders should organize the data as follows in `Bengali-Hateful-Memes/data`,

```
├── coco
│   └── train2017
├── gqa
│   └── images
├── ocr_vqa
│   └── images
├── textvqa
│   └── train_images
└── vg
    ├── VG_100K
    └── VG_100K_2
```
