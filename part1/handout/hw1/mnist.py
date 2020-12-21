"""Problem 3 - Training on MNIST"""
import numpy as np

# TODO: Import any mytorch packages you need (XELoss, SGD, etc)
from mytorch.optim.sgd import SGD
from mytorch.nn.functional import cross_entropy
from mytorch.nn.sequential import Sequential
from mytorch.nn.linear import Linear
from mytorch.nn.activations import ReLU
from mytorch.tensor import Tensor


# NOTE: Batch size pre-set to 100. Shouldn't need to change.
BATCH_SIZE = 100

def mnist(train_x, train_y, val_x, val_y):
    """Problem 3.1: Initialize objects and start training
    You won't need to call this function yourself.
    (Data is provided by autograder)
    
    Args:
        train_x (np.array): training data (55000, 784) 
        train_y (np.array): training labels (55000,) 
        val_x (np.array): validation data (5000, 784)
        val_y (np.array): validation labels (5000,)
    Returns:
        val_accuracies (list(float)): List of accuracies per validation round
                                      (num_epochs,)
    """
    # TODO: Initialize an MLP, optimizer, and criterion
    model = Sequential(Linear(784, 20), ReLU(), Linear(20, 10))
    optimizer = SGD(model.parameters(), lr=0.1)
    criterion = cross_entropy

    # TODO: Call training routine (make sure to write it below)
    val_accuracies = train(model, optimizer, criterion, train_x, train_y, val_x, val_y)

    return val_accuracies

def shuffle_train_data(train_x, train_y):
    ind_list = list(range(train_x.shape[0]))
    np.random.shuffle(ind_list)
    return train_x[ind_list], train_y[ind_list]

def split_data_into_batches(input, target, batch_size = 100):
    batches = []
    total_size = input.shape[0]
    batch_length = total_size // batch_size + (total_size % batch_size > 0)
    for i in range(batch_length):
        batches.append((input[i * batch_size : (i+1) * batch_size],
                        target[i * batch_size : (i+1) * batch_size]))

    return batches



def train(model, optimizer, criterion, train_x, train_y, val_x, val_y, num_epochs=3):
    """Problem 3.2: Training routine that runs for `num_epochs` epochs.
    Returns:
        val_accuracies (list): (num_epochs,)
    """
    model.train()
    val_accuracies = []

    for epoch in range(num_epochs):
        shuffle_train_x, shuffle_train_y = shuffle_train_data(train_x, train_y)
        batches = split_data_into_batches(shuffle_train_x, shuffle_train_y)
        for i, (batch_data, batch_labels) in enumerate(batches):
            optimizer.zero_grad()
            out = model.forward(Tensor(batch_data))
            loss = criterion(out, Tensor(batch_labels))
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                accuracy = validate(model, val_x, val_y)
                val_accuracies.append(accuracy)
                model.train()

    return val_accuracies


def validate(model, val_x, val_y):
    """Problem 3.3: Validation routine, tests on val data, scores accuracy
    Relevant Args:
        val_x (np.array): validation data (5000, 784)
        val_y (np.array): validation labels (5000,)
    Returns:
        float: Accuracy = correct / total
    """
    #TODO: implement validation based on pseudocode
    model.eval()

    batches = split_data_into_batches(val_x, val_y)
    num_correnct = 0
    for batch_data, batch_labels in batches:
        out = model.forward(Tensor(batch_data))
        batch_preds = np.argmax(out.data, axis=1)
        num_correnct += (batch_labels == batch_preds).sum()

    return num_correnct / len(val_x)
