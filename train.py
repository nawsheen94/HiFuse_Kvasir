import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
import os
from models.HiFuseSmall import HiFuseSmall
import torch.nn as nn
import torch.optim as optim
import zipfile

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Check if the datasets directory exists and if not, create it
if not os.path.exists('datasets/Kvasir'):
    os.makedirs('datasets/Kvasir')

# Unzip the dataset, handling potential file conflicts
zip_path = '/content/drive/MyDrive/Kvasir.zip'
extract_path = 'datasets/'  # Extract directly into 'datasets'

def create_dirs(extract_path, member):
    """Create directories if they don't exist"""
    path = os.path.join(extract_path, member.filename)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    for member in zip_ref.infolist():
        create_dirs(extract_path, member)
        try:
            zip_ref.extract(member, extract_path)
        except zipfile.BadZipFile as e:
            print(f"Error extracting {member.filename}: {e}")

# Load dataset
dataset = ImageFolder(root='/content/data/KVasir', transform=transform)
train_size = int(0.5 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize model, loss function, and optimizer
num_classes = len(dataset.classes)
model = main_model(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training and validation loops with TensorBoard logging
log_dir = os.path.join("runs", "experiment1")
writer = SummaryWriter(log_dir)

num_epochs = 5  # Reduced number of epochs for quicker training

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs, writer):
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = val_loader
            
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = running_corrects.double() / len(data_loader.dataset)

            writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
            writer.add_scalar(f'Accuracy/{phase}', epoch_acc, epoch)

train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs, writer)
writer.close()
