import sys
sys.path.append("/Users/weins/airflow/dags/ml_pipeline")

import torch
import s3_storage
from neural_network import NeuralNetwork
from device import device
from datetime import datetime
from torch import nn
from torch.utils.data import DataLoader

# Get cpu, gpu or mps device for training.
print(f"Using {device} device")

def train():
    # Get the data from S3
    todays_date = datetime.today().strftime('%Y-%m-%d')
    training_data_buffer = s3_storage.download_from_s3(f'training_data_{todays_date}')
    training_data = torch.load(training_data_buffer)

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

    # Upload model to s3
    s3_storage.upload_to_s3(model, f'model_{todays_date}')

# Call train function
train()