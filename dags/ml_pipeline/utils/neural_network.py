from torch import nn

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        # initializes the neural network with the constructor
        super().__init__()

        # prepare the nn for accepting our 28x28 pixel image
        self.flatten = nn.Flatten()

        # define the layers of the model
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512), # defines linear layer with 28*28=784 features that outputs 512 features
            nn.ReLU(), # Rectified linear unit (non-linear function applied, max(0, x) where x is some normalized value between 0.0 and 1.0)
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10) # output only 10 features so we can use these to map back to our 10 clothing categories (see 'image_classes.py')
        )

    # Defines how to pass data through the network
    def forward(self, x):
        # flattens an image tensor
        x = self.flatten(x)
        # passes data through the network
        logits = self.linear_relu_stack(x)
        return logits