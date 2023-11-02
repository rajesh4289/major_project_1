import torch

from models import model

def calculate_accuracy(model, dataloader):
    model.eval()  # Set the model in evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data['image'], data['label']
            outputs = model(inputs)
            
            # Assuming a threshold of 0.5 for multi-label classification
            predicted_labels = (outputs > 0.5).float()
            
            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy

# Example usage:
# Calculate accuracy on training, validation, and test datasets
train_accuracy = calculate_accuracy(model, train_dataloader)
valid_accuracy = calculate_accuracy(model, validation_dataloader)
test_accuracy = calculate_accuracy(model, test_dataloader)

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Validation Accuracy: {valid_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
