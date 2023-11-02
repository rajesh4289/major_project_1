import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import models
from engine import train, validate
from dataset import ImageDataset
from torch.utils.data import DataLoader

# Initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Initialize the model
model = models.model(pretrained=True, requires_grad=False).to(device)

# Define the learning parameters
lr = 0.0001
epochs = 20
batch_size = 32
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss()

# Read the training CSV file
train_csv = pd.read_csv('E:/Major Project/input/Multi_Label_dataset/train.csv')

# Create the training dataset (remove the 'transform' argument)
train_dataset = ImageDataset(csv=train_csv, train=True, test=False)

# Create the training data loader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Create a list to store training loss and validation loss
train_loss = []
valid_loss = []

# Training loop
for epoch in range(epochs):
    print(f"Epoch {epoch + 1} of {epochs}")

    # Train the model
    train_epoch_loss = train(model, train_loader, optimizer, criterion, train_dataset, device)

    # Print and store training loss
    print(f"Train Loss: {train_epoch_loss:.4f}")
    train_loss.append(train_epoch_loss)

# Save the trained model to disk
torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': criterion,
}, 'E:/Major Project/input/outputs/model.pth')

# Plot and save the train loss
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('E:/Major Project/input/outputs/train_loss.png')
plt.show()
