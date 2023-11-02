import models
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from engine import validate
from dataset import ImageDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# Initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model
model = models.model(pretrained=False, requires_grad=False).to(device)

# Load the model checkpoint
# Load the model checkpoint with map_location
checkpoint = torch.load('outputs/model.pth', map_location=torch.device('cpu'))


# Load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Read the test csv file
test_csv = pd.read_csv('Multi_Label_dataset/test.csv')

# Initialize the test dataset
# Initialize the test dataset for testing
test_data = ImageDataset(test_csv, train=False, test=True)


# Initialize the test data loader
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

def calculate_accuracy(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in dataloader:
            images, targets = data['image'].to(device), data['label'].to(device)
            outputs = model(images)
            outputs = (torch.sigmoid(outputs) > 0.5).float()  # Threshold outputs
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

# Calculate test accuracy
test_acc = calculate_accuracy(model, test_loader, device)
print(f'Test Accuracy: {test_acc:.4f}')

# Plot and save the test accuracy
plt.figure(figsize=(10, 7))
plt.plot(test_acc, color='red', label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('outputs/test_accuracy.png')
plt.show()