import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import torch.optim as optim
from torch.optim import lr_scheduler
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from madgrad import MADGRAD
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
import clip
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("Device:",device)


# Define the MultiheadAttention class
class MultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def forward(self, query, key, value, mask=None):
        output, _ = self.attention(query, key, value, attn_mask=mask)
        return output


clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model = clip_model.visual

# Convert model's weights to single-precision
clip_model = clip_model.float()

# Create an instance of the CLIP model
clip_model = clip_model.to(device)

# Freeze the parameters of the CLIP model
for param in clip_model.parameters():
    param.requires_grad = False   



# Define the model in PyTorch
class DORA(nn.Module):
    def __init__(self, clip_model, num_classes, num_heads):
        super(DORA, self).__init__()

        # Visual feature extractor (CLIP)
        self.clip = clip_model # Load the CLIP model
        self.visual_linear = nn.Linear(512, 1024)

        # Textual feature extractor (BERT)
        self.bert = AutoModel.from_pretrained("facebook/xglm-564M")

        # Multihead attention
        self.attention = MultiheadAttention(d_model=1024, nhead=num_heads)

        # Fully connected layers (updated for binary classification)
        self.fc = nn.Sequential(
            nn.Linear(1024 + 1024 + 1024 + 1024, 32),  # Input size includes visual features, BERT embeddings, and attention output
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(32, num_classes),  # Updated for binary classification
            nn.Sigmoid()  # Updated for binary classification
        )

    def forward(self, image_input, input_ids, attention_mask):

        # Extract visual features using CLIP
        image_features = self.clip(image_input)
        image_features = self.visual_linear(image_features)
        image_features = image_features.unsqueeze(1)
        # Apply average pooling to reduce the sequence length to 50
        image_features = F.adaptive_avg_pool1d(image_features.permute(0, 2, 1), 50).permute(0, 2, 1)
        # print(image_features.shape)


        # Extract BERT embeddings
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_output = bert_outputs.last_hidden_state
        # print(bert_output.shape)

        # Apply multihead attention between visual_features and BERT embeddings
        # Assuming that visual_features and bert_output have shape (seq_length, batch_size, feature_size)
        attention_output1 = self.attention(
            query=image_features.permute(1, 0, 2),  # Swap batch_size and seq_length dimensions
            key=bert_output.permute(1, 0, 2),  # Swap batch_size and seq_length dimensions
            value=bert_output.permute(1, 0, 2),  # Swap batch_size and seq_length dimensions
            mask=None  # You can add a mask if needed
        )


        # Swap back the dimensions to (batch_size, seq_length, feature_size)
        attention_output1 = attention_output1.permute(1, 0, 2)
        # print(attention_output.shape)

        # Assuming that visual_features and bert_output have shape (seq_length, batch_size, feature_size)
        attention_output2 = self.attention(
            query=image_features.permute(1, 0, 2),  # Swap batch_size and seq_length dimensions
            key=bert_output.permute(1, 0, 2),  # Swap batch_size and seq_length dimensions
            value=image_features.permute(1, 0, 2),  # Swap batch_size and seq_length dimensions
            mask=None  # You can add a mask if needed
        )


        # Swap back the dimensions to (batch_size, seq_length, feature_size)
        attention_output2 = attention_output2.permute(1, 0, 2)
        # print(attention_output.shape)

        # Concatenate the context vector, visual features, BERT embeddings, and attention output
        fusion_input = torch.cat([attention_output1, attention_output2, image_features, bert_output], dim=2)
        # print(fusion_input.shape)

        output = self.fc(fusion_input.mean(1))  # Pool over the sequence dimension
        return output


# Define a function to calculate accuracy for task 1
def calculate_accuracy_task1(predictions, targets):
    predictions = (predictions > 0.5).float()
    correct = (predictions == targets).float()
    accuracy = correct.sum() / len(correct)
    return accuracy


# Define a function to calculate accuracy for multiclass
def calculate_accuracy_task2(predictions, targets):
    # For multi-class classification, you can use torch.argmax to get the predicted class
    predictions = torch.argmax(predictions, dim=1)
    correct = (predictions == targets).float()
    accuracy = correct.sum() / len(correct)
    return accuracy


# num_epochs = 1

def train(train_loader, val_loader, task, path, heads, class_weights, epochs, lr_rate):

  if task == 'task2':
    num_classes = 4  # Number of output classes (target classification)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
  else:
    num_classes = 1  
    criterion = nn.BCELoss()

  num_heads = heads  # Number of attention heads for multihead attention
  model = DORA(clip_model, num_classes, num_heads)
  model = model.to(device)   

  # Define loss and optimizer
  
  # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  optimizer = MADGRAD(model.parameters(), lr=lr_rate)

  num_epochs = epochs
  num_training_steps = num_epochs * len(train_loader)
  lr_scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
  )
    # Training loop
  best_val_accuracy = 0.0

  print(f"Start Training DORA for {task}")
  print("--------------------------------")
  print("Number of Heads:",heads)
  print("Epochs:",epochs)
  print("Learning Rate:",lr_rate )
  for epoch in range(num_epochs):
      model.train()
      total_loss = 0
      total_accuracy = 0

      # Wrap the train_loader with tqdm for the progress bar
      with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as t:
          for batch in t:
              images = batch['image'].to(device)
              input_ids = batch['input_ids'].to(device)
              attention_mask = batch['attention_mask'].to(device)
              if task == 'task2':
                labels = batch['label'].to(device)
              else:
                labels = batch['label'].float().to(device)


              optimizer.zero_grad()
              outputs = model(images, input_ids, attention_mask)
              if task == 'task1':
                outputs = outputs.squeeze(dim=1)

              loss = criterion(outputs, labels)
              loss.backward()
              optimizer.step()
              total_loss += loss.item()
              if task == 'task1':
                total_accuracy += calculate_accuracy_task1(outputs, labels).item()
              else:
                total_accuracy += calculate_accuracy_task2(outputs, labels).item()

              # Update the tqdm progress bar
              t.set_postfix(loss=total_loss / (t.n + 1), acc=total_accuracy / (t.n + 1))

      # Calculate training accuracy and loss
      avg_train_loss = total_loss / len(train_loader)
      avg_train_accuracy = total_accuracy / len(train_loader)

      # Validation loop
      model.eval()
      val_labels = []
      val_preds = []
      total_val_loss = 0  # Variable to accumulate validation loss
      with torch.no_grad():
          for batch in tqdm(val_loader, desc=f"Validation", unit="batch"):
              images = batch['image'].to(device)
              input_ids = batch['input_ids'].to(device)
              attention_mask = batch['attention_mask'].to(device)
              if task == 'task2':
                labels = batch['label'].to(device)
              else:
                labels = batch['label'].float().to(device)

              # outputs = model(images, input_ids, attention_mask).squeeze().cpu().numpy()
              outputs = model(images, input_ids, attention_mask)
              if task == 'task1':
                outputs = outputs.squeeze(dim=1)

              val_loss = criterion(outputs, labels)  # Calculate validation loss
              total_val_loss += val_loss.item()
              if task == 'task1':
                preds = (outputs > 0.5).float()  # binary
              else:
                preds = torch.argmax(outputs, dim=1).cpu().numpy() # multiclass 

              val_labels.extend(labels.cpu().numpy())
              if task == 'task1':
                val_preds.extend(preds.cpu().numpy())
              else:
                val_preds.extend(preds)  

      # Calculate validation accuracy and loss
      val_accuracy = accuracy_score(val_labels, val_preds)
      avg_val_loss = total_val_loss / len(val_loader)
      print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_accuracy * 100:.2f}%,  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy * 100:.2f}%")

      # Save the model with the best validation accuracy
      if val_accuracy > best_val_accuracy:
          best_val_accuracy = val_accuracy
          torch.save(model.state_dict(), os.path.join(path, f'model_{task}.pth'))
          print("Model Saved.")

      lr_scheduler.step()  # Update learning rate

  print(f"Best Validation Accuracy: {best_val_accuracy * 100:.2f}%")
  print("--------------------------------")
  return model


def evaluation(path,task, model, test_loader):
  # Load the saved model
  # model = MultimodalModel(bert_model, resnet_model).to(device)
  print("Model is Loading..")
  model.load_state_dict(torch.load(os.path.join(path, f'model_{task}.pth')))
  model.eval()
  print("Loaded.")

  test_labels = []
  test_preds = []

  print("--------------------------------")
  print("Start Evaluating..")
  # Use tqdm for the test data
  with torch.no_grad(), tqdm(test_loader, desc="Testing", unit="batch") as t:
      for batch in t:
          images = batch['image'].to(device)
          input_ids = batch['input_ids'].to(device)
          attention_mask = batch['attention_mask'].to(device)
          labels = batch['label'].float().to(device)

          
          if task == 'task1':
            outputs = model(images, input_ids, attention_mask).squeeze().cpu().numpy()
            preds = (outputs > 0.5).astype(int)
          else:  
            outputs = model(images, input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

          test_labels.extend(labels.cpu().numpy())
          test_preds.extend(preds)

  print('Evaluation Done.')
  print("--------------------------------")
  return test_labels, test_preds




