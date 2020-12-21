import os
import sys
import numpy as np

from mytorch.nn.activations import Tanh, ReLU, Sigmoid
from mytorch.nn.conv import Conv1d, Flatten
from mytorch.nn.functional import get_conv1d_output_size
from mytorch.nn.linear import Linear
from mytorch.nn.module import Module
from mytorch.nn.sequential import Sequential


class CNN(Module):
    """A simple convolutional neural network with the architecture described in Section 3.
    
    You'll probably want to implement `get_final_conv_output_size()` on the
        bottom of this file as well.
    """
    def __init__(self):
        super().__init__()
        
        # You'll need these constants for the first layer
        first_input_size = 60 # The width of the input to the first convolutional layer
        first_in_channel = 24 # The number of channels input into the first layer
        
        # TODO: initialize all layers EXCEPT the last linear layer
        layers = [
            Conv1d(out_channel=56, in_channel=first_in_channel, kernel_size=5, stride=1),
            Tanh(),
            Conv1d(out_channel = 28, in_channel = 56, kernel_size=6, stride=2),
            ReLU(),
            Conv1d(out_channel=14, in_channel=28, kernel_size=2, stride=2),
            Sigmoid(),
            Flatten(),
            # ... etc ... put layers in here, comma separated
        ]
        
        # TODO: Iterate through the conv layers and calculate the final output size
        final_output_size = get_final_conv_output_size(layers, first_input_size)
        # TODO: Append the linear layer with the correct size onto `layers`
        layers.append(Linear(in_features=14 * final_output_size, out_features=10))

        # TODO: Put the layers into a Sequential
        self.layers = Sequential(*layers)
        

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channels, input_size)
        Return:
            out (np.array): (batch_size, out_feature)
        """
        # Already completed for you. Passes data through all layers in order.
        return self.layers(x)

def get_final_conv_output_size(layers, input_size):
    """Calculates how the last dimension of the data will change throughout a CNN model
    
    Note that this is the final output size BEFORE the flatten.
    
    Note:
        - You can modify this function to use in HW2P2.
        - If you do, consider making these changes:
            - Change `layers` variable to `model` (a subclass of `Module`),
                and iterate through its submodules
                - Make a toy model in torch first and learn how to do this
            - Modify calculations to account for other layer types (like `Linear` or `Flatten`)
            - Change `get_conv1d_output_size()` to account for stride and padding (see its comments)
    
    Args:
        layers (list(Module)): List of Conv1d layers, activations, and flatten layers
        input_size (int): input_size of x, the input data 
    """
    # Hint, you may find the function `isinstance()` to be useful.
    output_size = input_size
    for l in layers:
        if type(l) == Conv1d:
            output_size = get_conv1d_output_size(input_size=output_size, kernel_size=l.kernel_size, stride=l.stride)
    return output_size
    # raise NotImplementedError("TODO: Complete get_final_conv_output_size()!")
