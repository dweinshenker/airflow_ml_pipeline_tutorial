import sys
sys.path.append("../ml_pipeline")

import torch
import random
from image_classes import classes
from datetime import datetime
from matplotlib import pyplot
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets 
from torchvision.transforms import ToTensor
from device import device
from neural_network import NeuralNetwork

# Get training data from open datasets
# This downloads the raw data into a the directory /data/FashionMNIST/raw
# Performs a transformation on 28x28 images to Tensors
# Automatically 
training_data = datasets.FashionMNIST(
    root="data", # directory where data should be downloaded to on local
    train=True, # if True, uses training data set of 60K images, False uses 10K set of test data images
    download=True, # if True downloades from internet, False only fetches from local
    transform=ToTensor() # converts image data to Tensor (multi-dimensional matrix)
)

# Get test data from open datasets
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# Utilize a data loader to iterate over datasets
batch_size = 128 # defines the number of samples processed per iteration

training_dl = DataLoader(training_data, batch_size=batch_size)
test_dl = DataLoader(test_data, batch_size=batch_size)

# print 1 batch of data for debugging
#   X: input example with dimensions [N, C, H, W]
#   N: number of samples
#   C: count of images per item samples
#   H: height of image (in pixels)
#   W: widgth of image (in pixels)
#   N: number of samples
#   y: label of sample with dimensions [N]
for X, y in training_dl:
    print(f"Shape of input example 'X' [N, C, H, W]: {X.shape}-{X.dtype}")
    print(f"Shape of label example 'y' [N]: {y.shape}-{y.dtype}")
    break

# Show what some of the images look like
for i in range(9):
    # define subplot
    pyplot.subplot(330 + 1 + i)
    # plot raw pixel data
    pyplot.imshow(training_data[i][0][0], cmap=pyplot.get_cmap('gray'))
# show the figure
pyplot.show()

# Get cpu, gpu or mps device for training.
print(f"Using {device} device")

# Utilize a data loader to iterate over datasets
batch_size = 128 # defines the number of samples processed per iteration

training_dl = DataLoader(training_data, batch_size=batch_size)
for X, y in training_dl:
    print(f"Shape of input example 'X' [N, C, H, W]: {X.shape}-{X.dtype}")
    print(f"Shape of label example 'y' [N]: {y.shape}-{y.dtype}")
    break

# Define model using:
# Model => NeuralNetwork
# Loss Function => Cross Entropy Loss (see https://365datascience.com/tutorials/machine-learning-tutorials/cross-entropy-loss/)
# Optimizer => Stochastic Gradient Descent (see https://www.geeksforgeeks.org/ml-stochastic-gradient-descent-sgd/)
#   - has a learning rate (how large of a jump to take in the direction of the gradient)
#   - backpropagation is the way in which loss is passed backwards against the gradients of the loss in the network
model = NeuralNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

# Run training epochs
epochs = 10 # epoch is a full run over the data
for t in range(epochs):
    print(f'Epoch {t+1}\n---------------------------')
    model.train() # sets the model to training mode
    size = len(training_dl.dataset)

    # Walk over training data in batches
    for batch, (X, y) in enumerate(training_dl):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss, current = loss.item(), (batch + 1) * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Load test data into DataLoader
test_dl = DataLoader(test_data, batch_size=batch_size)
for X, y in test_dl:
    print(f"Shape of input example 'X' [N, C, H, W]: {X.shape}-{X.dtype}")
    print(f"Shape of label example 'y' [N]: {y.shape}-{y.dtype}")
    break

size = len(test_dl.dataset)
num_batches = len(test_dl)
model.eval()
loss_fn = nn.CrossEntropyLoss()
test_loss, correct = 0, 0
with torch.no_grad():
    for X, y in test_dl:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        test_loss += loss_fn(pred, y).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
test_loss /= num_batches
correct /= size
print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# look at some of the predictions
for idx in range(9):
    # define subplot
    pyplot.subplot(330 + 1 + idx)
    # plot raw pixel data, get random value
    i = random.randint(0, len(test_data)-1)
    pyplot.imshow(test_data[i][0][0], cmap=pyplot.get_cmap('gray'))
    # prediction
    x, y = test_data[i][0], test_data[i][1]
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')
        pyplot.title(f'Predicted: "{predicted}", Actual: "{actual}"')

# show the figure
pyplot.show()