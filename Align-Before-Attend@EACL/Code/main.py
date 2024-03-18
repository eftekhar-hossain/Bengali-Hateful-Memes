import sys
import pandas as pd
import pickle as pkl
import warnings
warnings.filterwarnings('ignore')
from dataset import ImgToArray,TextTokenizer
from models import MultimodalContextualFusion

def main():
    arguments = sys.argv[1:]

    # Check if '-d-img' is provided
    if '-d-img' in arguments:

        img_dir = arguments[arguments.index('-img_dir') + 1]
        # print(arguments[arguments.index('-train') + 1])
        train = pd.read_excel(arguments[arguments.index('-train') + 1]+'.xlsx')
        valid = pd.read_excel(arguments[arguments.index('-valid') + 1]+'.xlsx')
        test = pd.read_excel(arguments[arguments.index('-test') + 1]+'.xlsx')
        column_name = arguments[arguments.index('-img_column_name') + 1]
        image_size = int(arguments[arguments.index('-img_size') + 1])
        dataset_name = arguments[arguments.index('-dataset_name') + 1]
        split_name = arguments[arguments.index('-split_name') + 1]
        split_name = split_name[1:-1].split(',')

        # print(type(split_name))

        ImgToArray(img_dir, train, valid, test, column_name, image_size, dataset_name, split_name)

    elif '-d-txt' in arguments:

        train = pd.read_excel(arguments[arguments.index('-train') + 1]+'.xlsx')
        valid = pd.read_excel(arguments[arguments.index('-valid') + 1]+'.xlsx')
        test = pd.read_excel(arguments[arguments.index('-test') + 1]+'.xlsx')
        txt_len = int(arguments[arguments.index('-txt_len') + 1])
        column_name = arguments[arguments.index('-txt_column_name') + 1]
        dataset_name = arguments[arguments.index('-dataset_name') + 1]
        split_name = arguments[arguments.index('-split_name') + 1]
        split_name = split_name[1:-1].split(',')

        
        TextTokenizer(train, valid, test, txt_len, column_name, dataset_name, split_name)   


    elif '-models' in arguments:

      method_name = arguments[arguments.index('-method_name') + 1]   
      datasets = arguments[arguments.index('-datasets') + 1]
      datasets = datasets[1:-1].split(',')
      train_excel = pd.read_excel(datasets[0]+'.xlsx')
      valid_excel = pd.read_excel(datasets[1]+'.xlsx')
      test_excel = pd.read_excel(datasets[2]+'.xlsx')
      

      train_pkl = arguments[arguments.index('-train_pkl') + 1]
      train_pkl = train_pkl[1:-1].split(',')
      valid_pkl = arguments[arguments.index('-valid_pkl') + 1]
      valid_pkl = valid_pkl[1:-1].split(',')
      test_pkl = arguments[arguments.index('-test_pkl') + 1]
      test_pkl = test_pkl[1:-1].split(',')
      label_column = arguments[arguments.index('-label_column') + 1]

      class_names = list(train_excel[label_column].value_counts().keys())
      label_dict = dict(zip(class_names,range(len(class_names))))

      ## Replace Names
      train_excel['enc_label'] = train_excel[label_column].replace(label_dict)
      valid_excel['enc_label'] = valid_excel[label_column].replace(label_dict)
      test_excel['enc_label'] = test_excel[label_column].replace(label_dict)


      ## Train image
      with open('Datasets/'+f'{train_pkl[0]}.pkl','rb') as f:
        train_image = pkl.load(f)
      ## Train text
      with open('Datasets/'+f'{train_pkl[1]}.pkl','rb') as f:
        train_text = pkl.load(f)
      ## Valid image
      with open('Datasets/'+f'{valid_pkl[0]}.pkl','rb') as f:
        valid_image = pkl.load(f)
      ## Valid text
      with open('Datasets/'+f'{valid_pkl[1]}.pkl','rb') as f:
        valid_text = pkl.load(f)
      ## Test image
      with open('Datasets/'+f'{test_pkl[0]}.pkl','rb') as f:
        test_image = pkl.load(f)
      ## Test text
      with open('Datasets/'+f'{test_pkl[1]}.pkl','rb') as f:
        test_text = pkl.load(f)

      train, valid, test = [train_text,train_image], [valid_text, valid_image], [test_text,test_image]  
      labels = [train_excel['enc_label'],valid_excel['enc_label'],test_excel['enc_label']]
        
      hparams = arguments[arguments.index('-hparams') + 1]
      hparams = eval(hparams)
      md_hparams = arguments[arguments.index('-md_hparams') + 1]
      md_hparams = eval(md_hparams)

    
      MultimodalContextualFusion(train,valid,test, labels, hparams, md_hparams, class_names, method_name)    

  

    else:
        print("ERROR: Argument not Available.")

if __name__ == "__main__":
    main()
