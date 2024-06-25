import os
import zipfile
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset

def unzip_dataset(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

def load_dataset(dataset_dir, transform):
    dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)
    return dataset

def get_data_loaders(dataset_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = load_dataset(dataset_dir, transform)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = dataset_size // 2

    fold1_indices = indices[:split]
    fold2_indices = indices[split:]

    fold1_dataset = Subset(dataset, fold1_indices)
    fold2_dataset = Subset(dataset, fold2_indices)

    train_loader_fold1 = DataLoader(fold1_dataset, batch_size=batch_size, shuffle=True)
    val_loader_fold1 = DataLoader(fold2_dataset, batch_size=batch_size, shuffle=False)
    train_loader_fold2 = DataLoader(fold2_dataset, batch_size=batch_size, shuffle=True)
    val_loader_fold2 = DataLoader(fold1_dataset, batch_size=batch_size, shuffle=False)

    return (train_loader_fold1, val_loader_fold1), (train_loader_fold2, val_loader_fold2)
