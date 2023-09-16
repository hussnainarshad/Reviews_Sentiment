

"""**5 Epocs**"""

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AlbertTokenizer, AlbertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from tqdm import tqdm
import time

# Load the dataset from a CSV file (replace with your dataset file path)
data = pd.read_csv('data.csv')

# Define labels and mapping
label_to_id = {'happy': 0, 'sad': 1, 'angry': 2, 'confused': 3, 'neutral': 4}
id_to_label = {v: k for k, v in label_to_id.items()}

# Tokenizer
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

# Data preprocessing
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        review = row['review']
        label = row['Label']

        encoding = self.tokenizer(review, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label_to_id[label], dtype=torch.long)
        }

# Split the dataset
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Create dataloaders
train_dataset = CustomDataset(train_data, tokenizer, max_length=128)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = CustomDataset(test_data, tokenizer, max_length=128)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Model
model = AlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=len(label_to_id))

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
num_epochs = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    start_time = time.time()

    # Wrap the train_loader with tqdm for a progress bar
    train_iterator = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch in train_iterator:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    elapsed_time = time.time() - start_time
    remaining_time = elapsed_time * (num_epochs - epoch - 1)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, Elapsed Time: {elapsed_time:.2f}s, Remaining Time: {remaining_time:.2f}s")

# Evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label']

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
classification_rep = classification_report(all_labels, all_preds, target_names=[id_to_label[i] for i in range(len(label_to_id))])

# Calculate additional metrics
f1 = f1_score(all_labels, all_preds, average='weighted')
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')

# Confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("Classification Report:\n", classification_rep)
print("Confusion Matrix:\n", conf_matrix)

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import ElectraTokenizer, ElectraForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from tqdm import tqdm
import time

# Load the dataset from a CSV file (replace with your dataset file path)
data = pd.read_csv('data.csv')

# Define labels and mapping
label_to_id = {'happy': 0, 'sad': 1, 'angry': 2, 'confused': 3, 'neutral': 4}
id_to_label = {v: k for k, v in label_to_id.items()}

# Tokenizer
tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')

# Data preprocessing
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        review = row['review']
        label = row['Label']

        encoding = self.tokenizer(review, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label_to_id[label], dtype=torch.long)
        }

# Split the dataset
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Create dataloaders
train_dataset = CustomDataset(train_data, tokenizer, max_length=128)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = CustomDataset(test_data, tokenizer, max_length=128)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Model
model = ElectraForSequenceClassification.from_pretrained('google/electra-small-discriminator', num_labels=len(label_to_id))

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
num_epochs = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    start_time = time.time()

    # Wrap the train_loader with tqdm for a progress bar
    train_iterator = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch in train_iterator:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    elapsed_time = time.time() - start_time
    remaining_time = elapsed_time * (num_epochs - epoch - 1)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, Elapsed Time: {elapsed_time:.2f}s, Remaining Time: {remaining_time:.2f}s")

# Evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label']

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
classification_rep = classification_report(all_labels, all_preds, target_names=[id_to_label[i] for i in range(len(label_to_id))])

# Calculate additional metrics
f1 = f1_score(all_labels, all_preds, average='weighted')
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')

# Confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("Classification Report:\n", classification_rep)
print("Confusion Matrix:\n", conf_matrix)

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from tqdm import tqdm
import time

# Load the dataset from a CSV file (replace with your dataset file path)
data = pd.read_csv('data.csv')

# Define labels and mapping
label_to_id = {'happy': 0, 'sad': 1, 'angry': 2, 'confused': 3, 'neutral': 4}
id_to_label = {v: k for k, v in label_to_id.items()}

# Tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Data preprocessing
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        review = row['review']
        label = row['Label']

        encoding = self.tokenizer(review, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label_to_id[label], dtype=torch.long)
        }

# Split the dataset
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Create dataloaders
train_dataset = CustomDataset(train_data, tokenizer, max_length=128)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = CustomDataset(test_data, tokenizer, max_length=128)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Model
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(label_to_id))

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
num_epochs = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    start_time = time.time()

    # Wrap the train_loader with tqdm for a progress bar
    train_iterator = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch in train_iterator:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    elapsed_time = time.time() - start_time
    remaining_time = elapsed_time * (num_epochs - epoch - 1)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, Elapsed Time: {elapsed_time:.2f}s, Remaining Time: {remaining_time:.2f}s")

# Evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label']

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
classification_rep = classification_report(all_labels, all_preds, target_names=[id_to_label[i] for i in range(len(label_to_id))])

# Calculate additional metrics
f1 = f1_score(all_labels, all_preds, average='weighted')
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')

# Confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("Classification Report:\n", classification_rep)
print("Confusion Matrix:\n", conf_matrix)

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from tqdm import tqdm
import time

# Load the dataset from a CSV file (replace with your dataset file path)
data = pd.read_csv('data.csv')

# Define labels and mapping
label_to_id = {'happy': 0, 'sad': 1, 'angry': 2, 'confused': 3, 'neutral': 4}
id_to_label = {v: k for k, v in label_to_id.items()}

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Data preprocessing
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        review = row['review']
        label = row['Label']

        encoding = self.tokenizer(review, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label_to_id[label], dtype=torch.long)
        }

# Split the dataset
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Create dataloaders
train_dataset = CustomDataset(train_data, tokenizer, max_length=128)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = CustomDataset(test_data, tokenizer, max_length=128)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_to_id))

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
num_epochs = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    start_time = time.time()

    # Wrap the train_loader with tqdm for a progress bar
    train_iterator = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch in train_iterator:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    elapsed_time = time.time() - start_time
    remaining_time = elapsed_time * (num_epochs - epoch - 1)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, Elapsed Time: {elapsed_time:.2f}s, Remaining Time: {remaining_time:.2f}s")

# Evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label']

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
classification_rep = classification_report(all_labels, all_preds, target_names=[id_to_label[i] for i in range(len(label_to_id))])

# Calculate additional metrics
f1 = f1_score(all_labels, all_preds, average='weighted')
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')

# Confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("Classification Report:\n", classification_rep)
print("Confusion Matrix:\n", conf_matrix)

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from tqdm import tqdm
import time

# Load the dataset from a CSV file (replace with your dataset file path)
data = pd.read_csv('data.csv')

# Define labels and mapping
label_to_id = {'happy': 0, 'sad': 1, 'angry': 2, 'confused': 3, 'neutral': 4}
id_to_label = {v: k for k, v in label_to_id.items()}

# Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Data preprocessing
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        review = row['review']
        label = row['Label']

        encoding = self.tokenizer(review, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label_to_id[label], dtype=torch.long)
        }

# Split the dataset
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Create dataloaders
train_dataset = CustomDataset(train_data, tokenizer, max_length=128)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = CustomDataset(test_data, tokenizer, max_length=128)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(label_to_id))

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
num_epochs = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    start_time = time.time()

    # Wrap the train_loader with tqdm for a progress bar
    train_iterator = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch in train_iterator:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    elapsed_time = time.time() - start_time
    remaining_time = elapsed_time * (num_epochs - epoch - 1)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, Elapsed Time: {elapsed_time:.2f}s, Remaining Time: {remaining_time:.2f}s")

# Evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label']

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
classification_rep = classification_report(all_labels, all_preds, target_names=[id_to_label[i] for i in range(len(label_to_id))])

# Calculate additional metrics
f1 = f1_score(all_labels, all_preds, average='weighted')
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')

# Confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("Classification Report:\n", classification_rep)
print("Confusion Matrix:\n", conf_matrix)

