import spacy
import torch
from torchtext import data
import torch.nn as nn
from torch.utils.data import Dataset, random_split
from dataclean import data1
import pandas as pd
from torchtext.data import Field
import torch.optim as optim

class LSTMClassifer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1) if self.lstm.bidirectional else hidden[-1, :, :])
        return self.fc(hidden.squeeze(0))
    

class CustomDataset(Dataset):
    def __init__(self, x, y, text_field):
        self.x: pd.DataFrame = x
        self.y: pd.DataFrame  = y
        self.text_field: Field = text_field

    def __getitem__(self, index):
        # Return a dictionary with 'features' and 'label' as keys
        text_columns = ['subject', 'body', 'urls', 'Sender Name', 'Sender Email']
        combined_text = ' '.join(str(self.x.iloc[index][col]) for col in text_columns)
        processed_text = self.text_field.preprocess(combined_text)
        
        sample = {
            'text': processed_text,
            'label': torch.tensor(self.y.iloc[index], dtype=torch.float)
        }
        return sample

    def __len__(self):
        # Return the total number of samples
        return len(self.x)


nlp = spacy.load('en_core_web_sm')
TEXT = data.Field(include_lengths=True)
LABEL = data.LabelField(dtype=torch.float)
dataset = CustomDataset(
    data1[['subject', 'body', 'urls', 'Sender Name', 'Sender Email']], 
    data1['label'],
    TEXT
)
train_size = int(0.8 * len(data1))
test_size = len(data1) - train_size
train_data, test_data = random_split(dataset, [train_size, test_size])

def tokenize(text):
    return [token.text for token in nlp(text)]
TEXT.tokenize = tokenize
TEXT.build_vocab(train_data, max_size = train_size)
LABEL.build_vocab(train_data)

train_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, test_data),
    batch_size = 64,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5

model = LSTMClassifer(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)

optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = criterion.to(device)

def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

N_EPOCHS = 50
for epoch in range(N_EPOCHS):
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    print(f'Epoch: {epoch+1.02}, Train Loss:{train_loss:.3f}, Train Acc: {train_acc*100:.2f}%')