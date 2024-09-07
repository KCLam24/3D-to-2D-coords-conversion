import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import matplotlib.pyplot as plt

# Step 1: Define the LargeNPZDataset class
class LargeNPZDataset(Dataset):
    def __init__(self, npz_file, split='train'):
        self.data = np.load(npz_file)
        self.split = split
        self.images = self.data[f'X_{split}']
        self.labels = self.data[f'y_{split}']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Convert to tensor and add channel dimension
        label = torch.tensor(label, dtype=torch.long)  # Convert label to tensor
        return image, label

# Step 2: Instantiate the dataset and DataLoader
npz_file = 'dataset.npz'

# Load the training dataset
train_dataset = LargeNPZDataset(npz_file, split='train')

# Step 3: Define a function to plot random images
def plot_random_images(dataset, filename):
    # Select 4 random indices from the dataset
    indices = random.sample(range(len(dataset)), 4)
    
    # Extract images and labels for these indices
    images = []
    labels = []
    for idx in indices:
        image, label = dataset[idx]
        images.append(image.squeeze(0).numpy())  # Remove channel dimension and convert to numpy
        labels.append(label.item())
    
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    for i, ax in enumerate(axs.flat):
        ax.imshow(images[i], cmap='gray')
        ax.set_title(f'Label: {labels[i]}', fontsize=10)
        ax.axis('off')  # Hide the axes for a cleaner look

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(f'{filename}.png')
    plt.close(fig)

# Step 4: Plot random images from the training dataset
plot_random_images(train_dataset, 'random_train_images')

print("Random images have been plotted and saved as 'random_train_images.png'")

