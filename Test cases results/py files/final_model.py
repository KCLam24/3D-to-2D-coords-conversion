### Change Varience, same mean at 1,1, not log normal
# Import packages
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from timeit import default_timer as timer
from typing import Dict, List
from sklearn.model_selection import train_test_split
#import platform
from scipy.stats import multivariate_normal, lognorm, expon, powerlaw
from sklearn.utils import shuffle
import os 

# Import torchvision
import torch
import torch.nn as nn
import torchvision 
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim

# Check version
print(f'Pytorch version: {torch.__version__}')
print(f'Torchvision version: {torchvision.__version__}\n')

# setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device
if device == 'cuda':
    print(torch.cuda.get_device_name(0))
    gpu_memory_info = torch.cuda.get_device_properties(0)
    total_memory = gpu_memory_info.total_memory / (1024 ** 3)  # Convert to GB
    free_memory = torch.cuda.memory_reserved(0) / (1024 ** 3)  # Convert to GB
    used_memory = torch.cuda.memory_allocated(0) / (1024 ** 3)  # Convert to GB
    print(f"\nTotal GPU Memory: {total_memory:.2f} GB")
    print(f"Free GPU Memory: {free_memory:.2f} GB")
    print(f"Used GPU Memory: {used_memory:.2f} GB")
else:
    print("CUDA is not available.")
	

def plot_images(image1, image2, image3, image4, filename):
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    images = [image1, image2, image3, image4]
    titles = ['Sphere System 1', 'Sphere System 2', 'Ellipsoid System 1', 'Ellipsoid System 2']

    for idx, image in enumerate(images):
        # Determine the position in the 2x2 grid
        ax = axs[idx // 2, idx % 2]
        
        # Display the image
        ax.imshow(image, cmap='gray')
        
        # Set the predefined title for each subplot
        ax.set_title(titles[idx], fontsize=10)
        
        ax.axis('off')  # Hide the axes for a cleaner look

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f'{filename}.png')
    plt.close(fig)
    
def plot_loss_curves(results: Dict[str, List[float]],filename):
    """ Plots training curves of a result dictionary """
    # Get loss value of result dictionary(training and testing)
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    # Get accuracy values of the result dictionary (training and testing)
    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    #Figure out no of epochs
    epochs = range(len(results["train_loss"]))
    
    #Setup plot
    # Setup plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the loss
    ax[0].plot(epochs, loss, label="Train_loss")
    ax[0].plot(epochs, test_loss, label="Test_loss")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Epochs")
    ax[0].legend()

    # Plot the accuracy
    ax[1].plot(epochs, accuracy, label="Train_accuracy")
    ax[1].plot(epochs, test_accuracy, label="Test_accuracy")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("Epochs")
    ax[1].legend()

    # Save the figure
    plt.savefig(f'{filename}.png')
    plt.close(fig)



# takes in data and labels to transform into dataset
class LargeNPZDataset(Dataset):
    def __init__(self, npz_file, split='train'):
        # Load only metadata initially
        self.data = np.load(npz_file, mmap_mode='r')
        self.X = self.data[f'X_{split}']
        self.y = self.data[f'y_{split}']

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Load a single sample
        X_sample = self.X[idx]
        y_sample = self.y[idx]
        
        # Convert to tensors
        X_sample = torch.tensor(X_sample, dtype=torch.float32)
        y_sample = torch.tensor(y_sample, dtype=torch.long)
        
        return X_sample, y_sample

# Usage
train_dataset = LargeNPZDataset('dataset.npz', split='train')
val_dataset = LargeNPZDataset('dataset.npz', split='val')
        
# CNN model    
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 32 * 32, 2)

    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x

model = SimpleCNN()
model.to(device)
print('Model being used is SimpleCNN')

# create train_step()
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device=device):
    
    # Put model in train mode
    model.train()
    
    # setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    
    # Loop through data loader, data batch
    for batch, (X, y) in enumerate(dataloader):
        # Send data to the target device
        X, y = X.to(device).unsqueeze(1), y.to(device)

        #1. Forward pass
        y_pred = model(X) #output model logits
        
        #2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        #3. Optimizer zero grad
        optimizer.zero_grad()

        #4. Loss backward
        loss.backward()

        #5. Optimizer step
        optimizer.step()

        #6. Calculate accuracy metric
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class==y).sum().item()/len(y_pred) # total no correct divided by len of sample

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

# create test_step
def test_step(model:  torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device=device):
    # Put model in eval mode
    model.eval()

    #Setup test loss and test accuract values
    test_loss, test_acc = 0, 0

    # Turn on inderence mode
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            #send data to target device
            X, y = X.to(device).unsqueeze(1), y.to(device)

            #1. Forward pass
            test_pred_logits = model(X)

            #2. Calculate the loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            #3. Calculate the accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    #Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss /  len(dataloader)
    test_acc = test_acc /  len(dataloader)
    return test_loss, test_acc


# Create train function
#1. Create a train function that takes in varius model parameters + optimizer + dataloaders
def train(model:torch.nn.Module,
          train_data: torch.utils.data.DataLoader,
          test_data: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 10,
          device=device):

    #Create result dictionary
    results = {'train_loss': [],
               'train_acc': [],
               'test_loss': [],
               'test_acc': []}
    # Loop through training and testing steps for x number of epochs
    for epoch in range(epochs):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_data,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                          device=device)
        
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_data,
                                        loss_fn=loss_fn,
                                        device=device)
        #Print out what's happening
        print(f"Epoch: {epoch+1} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")

        #Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results


# testing begins
loss_name = 'Ultimate_loss2'

# Instantiate the dataset
BATCH_SIZE = 64
NUM_WORKERS = 0 #os.cpu_count()

train_dataset = LargeNPZDataset('dataset.npz', split='train')
val_dataset = LargeNPZDataset('dataset.npz', split='val')


# Create DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,num_workers=NUM_WORKERS)

# Trainig begins
# Set no of epochs (newnet)
NUM_EPOCHS = 500

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# FIrst test  without training
test_loss, test_acc = test_step(model=model,
                                dataloader=val_dataloader,
                                loss_fn=loss_fn,
                                device=device)
print(f" Model performance before training | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}\n")


# Start timer
start_time = timer()

# Train model
model_results = train(model=model,
                     train_data=train_dataloader,
                     test_data=val_dataloader,
                     optimizer=optimizer,
                     loss_fn=loss_fn,
                     epochs=NUM_EPOCHS)


# End timer and print out time taken
end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")

# Calculate parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'Total parameters: {total_params}\n')
plot_loss_curves(model_results, f'{loss_name}')
