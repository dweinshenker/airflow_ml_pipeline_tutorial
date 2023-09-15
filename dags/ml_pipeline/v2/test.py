import sys
sys.path.append("../ml_pipeline")

import torch
import random
import s3_storage
import metrics
from image_classes import classes
from device import device
from datetime import datetime
from matplotlib import pyplot
from torch import nn
from torch.utils.data import DataLoader

# Get cpu, gpu or mps device for training.
print(f"Using {device} device")

def test():
    # Get the data from S3
    todays_date = datetime.today().strftime('%Y-%m-%d')
    test_data_buffer = s3_storage.download_from_s3(f'test_data_{todays_date}')
    test_data = torch.load(test_data_buffer)

    # Get the model from S3
    model = s3_storage.download_from_s3(f'model_{todays_date}')

    # Load model
    model = torch.load(model)

    # Utilize a data loader to iterate over datasets
    batch_size = 128 # defines the number of samples processed per iteration

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

    # instrument accuracy metric to be exported
    metrics.generate_ml_metrics(100*correct)

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

# Call train function
test()