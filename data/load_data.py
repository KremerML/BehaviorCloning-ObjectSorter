import pandas as pd
import os
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

class CIFAR10HDataset(Dataset):
    def __init__(self, cifar10_dataset, dataframe, labels_probs, transform=None):
        self.cifar10_dataset = cifar10_dataset
        self.dataframe = dataframe
        self.labels_probs = labels_probs
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        cifar10_idx = row['cifar10_test_test_idx']

        image, _ = self.cifar10_dataset[cifar10_idx]
        if self.transform:
            image = self.transform(image)

        label_prob = self.labels_probs[cifar10_idx]

        return image, label_prob

def download_cifar10(root_dir):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    cifar10_dataset = datasets.CIFAR10(root=os.path.join(root_dir, 'cifar-10-images'), train=False, download=True, transform=transform)
    return cifar10_dataset

def load_data(csv_file, root_dir, probs_file, counts_file, num_classes=10):
    # Read the CIFAR-10H CSV file
    df = pd.read_csv(csv_file)
    df = df[df['cifar10_test_test_idx'] != -99999]  # Filter out attention check trials

    # Load probabilities and counts files
    labels_probs = np.load(probs_file)

    # Download CIFAR-10 dataset
    cifar10_dataset = download_cifar10(root_dir)

    # Splitting dataset
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Creating datasets
    train_dataset = CIFAR10HDataset(cifar10_dataset, train_df, labels_probs)
    test_dataset = CIFAR10HDataset(cifar10_dataset, test_df, labels_probs)

    # Creating data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader
