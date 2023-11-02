import torch
from torchtext.legacy.data.field import LabelField


import torch
from torchtext.legacy.data import Field
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.legacy import data
from torchtext.legacy.data import Field, TabularDataset

import spacy
import random
import numpy as np
TEXT = Field(tokenize='spacy', lower=True, include_lengths=True)
LABEL = Field(sequential=False, use_vocab=False, dtype=torch.float)

# Define the attention mechanism

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, embedded, hidden):
        # embedded: (src_len, batch_size, emb_dim)
        # hidden: (batch_size, hidden_dim)

        src_len = embedded.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((embedded, hidden), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)

# Define the text classification model with attention
class TextAttnModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):
        super(TextAttnModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.permute(1, 0, 2)
        outputs, (hidden, cell) = self.lstm(embedded)
        attention = self.attention(outputs, hidden[-1])
        weighted = torch.bmm(attention.unsqueeze(1), outputs.permute(1, 0, 2))
        weighted = weighted.squeeze(1)
        prediction = self.fc(weighted)
        return prediction

# Create fields for data preprocessing
TEXT = Field(tokenize='spacy', lower=True, include_lengths=True)
LABEL = data.LabelField(dtype=torch.float)

# Load your dataset (change the path and format)
fields = [('text', TEXT), ('label', LABEL)]
train_data, test_data = TabularDataset.splits(
    path='Multi_Label_dataset',
    train='train.csv',
    test='test.csv',
    format='csv',
    fields=fields
)

# Build vocabulary and load pre-trained word vectors
TEXT.build_vocab(train_data, max_size=25000, vectors='glove.6B.100d')
LABEL.build_vocab(train_data)

# Define device and data iterators
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
train_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, test_data),
    batch_size=BATCH_SIZE,
    sort_within_batch=True,
    sort_key=lambda x: len(x.text),
    device=device
)

# Define the model
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = TextAttnModel(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)
model.embedding.weight.data.copy_(TEXT.vocab.vectors)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

# Define a function for training
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.text
        predictions = model(text)
        loss = criterion(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# Define a function for evaluation
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text)
            loss = criterion(predictions, batch.label)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# Train the model
N_EPOCHS = 10
for epoch in range(N_EPOCHS):
    train_loss = train(model, train_iterator, optimizer, criterion)
    test_loss = evaluate(model, test_iterator, criterion)
    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\tTest Loss: {test_loss:.3f}')

# Save or use the trained model for predictions
