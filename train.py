import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
import os

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load dataset
dataset = ImageFolder(root='datasets/Kvasir', transform=transform)
train_size = int(0.5 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize model, loss function, and optimizer
num_classes = len(dataset.classes)
model = HiFuseSmall(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training and validation loops with TensorBoard logging
log_dir = os.path.join("runs", "experiment1")
writer = SummaryWriter(log_dir)

num_epochs = 5  # Reduced number of epochs for quicker training

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

writer.close()

# Load the TensorBoard notebook extension
%load_ext tensorboard

# Launch TensorBoard
%tensorboard --logdir runs
