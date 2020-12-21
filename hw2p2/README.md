# HW2P2 - Face Verification using Convolutional Neural Networks - Writeup

Kexin Zhang AndrewID: kexinzha

## Steps to run the model
1) python run.py
This script runs the training process and saves model file.
2) Python submit.py
This script generates the submission file. 

## DataLoader:
For training images and validation images, package torchvision.datasets.ImageFolder is used.
Verification data is loaded by the class VerifyDataset defined in dataloader.py. 
Submission test data is loaded by the class submitDataset defined in dataloader.py.
	1) __init__: read the txt file. Images are opened and transferred into tensors. All tensors are finally nconcatenated into one big tensor.
	2) __getitem__: return three elements: image1 tensor, image2 tensor and the label.

## Hyperparameters
(1) Batch Size = 256 for training data loader.  Batch size = 128 for validation and verification task data loader. 
(2) optimizer: 
SGD optimizer
	a) initial learning rate = 0.15
	b) momentum = 0.9
	c) weight decay = 5e-5
Adam optimizer
	a) initial learning rate = 1e-4
	b) weight decay = 5e-5
(3) scheduler:
MultiplcativeLR scheduler is used. Factor = 0.85
(4) num_workers = 4

## Training Process
(1) Train 12 epochs with SGD optimizer with initial learning rate 0.15. 
(2) Train 10 epochs with Adam optimizer with initial learning rate 1e-4.

## Network Architecture
The network architecture is defined in wider_baseline.py
It is a wider and deeper version of baseline model. There are MaxPool layers between convolution blocks. Within the convolution blocks, there are BatchNorm and ReLU layers.
The architecture: [3 * 3, 64] * 3, [3 * 3, 128] * 4, [3 * 3, 256] * 6, [3 * 3, 512] * 3, [3 * 3, 1024] * 1

CrossEntropy Loss is used as the criterion. 
Therefore, the embedding size is 1024. Then the embedding is passed into linear layers ([1024, 1024, 4000]) for the classification task.
