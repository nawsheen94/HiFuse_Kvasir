import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from models.HiFuseSmall import HiFuseSmall
from utils import unzip_dataset, get_data_loaders

# Hyperparameters
num_classes = 8
num_epochs = 10
batch_size = 32
learning_rate = 0.001

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Unzip dataset
unzip_dataset('/content/drive/MyDrive/Kvasir.zip', 'datasets/Kvasir')

# Get data loaders
(train_loader_fold1, val_loader_fold1), (train_loader_fold2, val_loader_fold2) = get_data_loaders('datasets/Kvasir', batch_size)

# Initialize model, loss function, and optimizer
model = HiFuseSmall(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# TensorBoard logging
writer = SummaryWriter('runs/HiFuse_Kvasir_experiment')

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs, writer, fold):
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

            writer.add_scalar(f'Fold{fold}/Loss/{phase}', epoch_loss, epoch)
            writer.add_scalar(f'Fold{fold}/Accuracy/{phase}', epoch_acc, epoch)

            print(f'Fold {fold} | Epoch {epoch}/{num_epochs - 1} | {phase} Loss: {epoch_loss:.4f} | {phase} Accuracy: {epoch_acc:.4f}')

# Train and validate on fold 1
train_and_validate(model, train_loader_fold1, val_loader_fold1, criterion, optimizer, num_epochs, writer, fold=1)

# Train and validate on fold 2
train_and_validate(model, train_loader_fold2, val_loader_fold2, criterion, optimizer, num_epochs, writer, fold=2)

writer.close()
