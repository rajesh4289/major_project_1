import models
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import tkinter as tk
from tkinter import filedialog

# Initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model
model = models.model(pretrained=False, requires_grad=False).to(device)

# Load the model checkpoint
checkpoint = torch.load('outputs/model.pth', map_location=device)

# Load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load the CSV file containing genre information
train_csv = pd.read_csv('Multi_Label_dataset/train.csv')
genres = train_csv.columns.values[2:]

# Define a transform to preprocess the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create a GUI window to browse for an image
root = tk.Tk()
root.withdraw()  # Hide the main window

# Ask the user to select an image
file_path = filedialog.askopenfilename()

if file_path:
    input_image = Image.open(file_path).convert('RGB')
    input_image = transform(input_image).unsqueeze(0).to(device)

    # Perform inference on the input image
    with torch.no_grad():
        outputs = model(input_image)
        outputs = torch.sigmoid(outputs)
        sorted_indices = np.argsort(outputs[0].cpu().numpy())
        best = sorted_indices[-3:]

    # Prepare the predicted genres string
    string_predicted = ''
    for i in range(len(best)):
        string_predicted += f"{genres[best[i]]}    "

    # Display the input image and predicted genres
    input_image = input_image.squeeze(0)
    input_image = input_image.permute(1, 2, 0).cpu().numpy()

    # Create a figure
    plt.figure()
    plt.imshow(input_image)
    plt.axis('off')
    plt.title(f"PREDICTED: {string_predicted}")

    # Ask the user to choose where to save the output image
    save_file_path = filedialog.asksaveasfilename(defaultextension=".png")

    if save_file_path:
        # Save the output image with the user-specified name and path
        plt.savefig(save_file_path, bbox_inches='tight', pad_inches=0)
        plt.show()

        print(f"Output image saved as {save_file_path}")
