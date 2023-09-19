# Prequisites
# Install Python
# Install Pytorch => https://pytorch.org/get-started/locally/

import sys
sys.path.append("/Users/weins/airflow/dags/ml_pipeline")

import torch
import utils.s3_storage
from datetime import datetime
from matplotlib import pyplot
from torch.utils.data import DataLoader # enables iterable access to input samples
from torchvision import datasets # available datasets
from torchvision.transforms import ToTensor # transforms data to tensors

def load():
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

    # upload data into s3 bucket
    todays_date = datetime.today().strftime('%Y-%m-%d')
    utils.s3_storage.upload_to_s3(training_data, f'training_data_{todays_date}')
    utils.s3_storage.upload_to_s3(test_data, f'test_data_{todays_date}')
    print("uploaded training + test data to S3")

# Call load function
load()