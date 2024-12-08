import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from PIL import ImageFile

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transforms: resizing images, converting to tensor, and normalizing with ImageNet stats
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.RandomHorizontalFlip(),  # Random horizontal flip for augmentation
    transforms.ToTensor(),  # Convert images to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Load datasets (adjust paths to your dataset)
train_data = datasets.ImageFolder(root=r'C:\Users\dhivy\Downloads\archive (1)\DATASET\TRAIN', transform=transform)
test_data = datasets.ImageFolder(root=r'C:\Users\dhivy\Downloads\archive (1)\DATASET\TEST', transform=transform)

# Create DataLoader objects
trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
testloader = DataLoader(test_data, batch_size=32, shuffle=False)

# Define a simple CNN model or use a pre-trained model (ResNet18 for transfer learning)
model = models.resnet18(pretrained=True)

# Freeze all layers except the final fully connected layer
for param in model.parameters():
    param.requires_grad = False

# Replace the final layer for our dataset's class count
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(train_data.classes))

# Move the model to the selected device
model = model.to(device)

# Define loss function and optimizer (only optimize the final layer)
optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)  # Fine-tuning with a lower learning rate

# Training function
def train_model(model, trainloader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in trainloader:
            if images is None:  # Skip if the image is corrupted
                continue

            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, labels)
            loss.backward()

            # Update weights
            optimizer.step()

            # Track loss and accuracy
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Print statistics for each epoch
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(trainloader):.4f}, Accuracy: {100 * correct / total:.2f}%")

# Evaluate the model
def evaluate_model(model, testloader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            if images is None:  # Skip if the image is corrupted
                continue

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# Training the model
epochs = 50
train_model(model, trainloader, nn.CrossEntropyLoss(), optimizer, epochs)

# Evaluate on test data
evaluate_model(model, testloader)

# Save the model
torch.save(model.state_dict(), 'model.pth')
