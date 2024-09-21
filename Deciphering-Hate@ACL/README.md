# How to Run

The following instructions will show how to run the **DORA** model on the **BHM** dataset. You can apply it to any multimodal (imag+text) dataset with slight modifications in the `dataset.py` file.

1. Clone this repository and navigate to Deciphering-Hate@ACL folder
```bash
git clone https://github.com/eftekhar-hossain/Bengali-Hateful-Memes.git
cd Bengali-Hateful-Memes
cd Deciphering-Hate@ACL
```

2. Install Packages and Dataset
```Shell
bash setup.sh  # Download the required libraries and dataset from the drive
```
If you already have the **BHM** folder in the current directory
```Shell
pip install -r requirements.txt  # download the required libraries
```
3. Directory
Folders should organized as follows in `Bengali-Hateful-Memes/Deciphering-Hate@ACL/`,

```
├── BHM
|   ├── Files
       ├── .xlsx
       └── .xlsx
|   └── Memes
       ├── .jpg
|      └── .png
├── Scripts
   ├──.py files
   └── .sh files      
   
```

3. Run the code

```bash
cd Scripts  # Set Scripts folder as the current directory 
```
```bash
bash run.sh  # run through bash in the terminal with default arguments 
```
Or
```bash
python main.py  # run directly in the terminal with default arguments 
```

Arguments:

- `--task`: binary (`task1`) or multiclass (`task2`) classification - `default: 'task1'`
- `--dataset`: the path of the dataset folder - `default: 'BHM'`
- `--max_len`: the maximum text length - `default: 50`
- `--batch_size`: `default: 50`
- `--model`: the saved models' path after training -`default: 'Saved_Models'`
- `--heads`: number of attention heads - `default: 2`
- `--n_iter`: number of epochs - `default: 1`
- `--lrate`: learning rate -  `default: 2e-5`



After calling the `main.py` or `run.sh`, you will see a classification report of the model performance on the test set of the respective task.
