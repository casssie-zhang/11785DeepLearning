"""This file contains new code for hw2 that you should copy+paste to the appropriate files.

We'll tell you where each method/class belongs."""


# ---------------------------------
# nn/functional.py
# ---------------------------------


    



# ---------------------------------
# nn/activations.py
# ---------------------------------

class Tanh(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.Tanh.apply(x)

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.Sigmoid.apply(x)
    
    
    