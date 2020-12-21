# HW3P2 - Utterance to Phoneme Mapping - Writeup


## Steps to run the model

python hw3p2_model3.py  

## DataLoader:  
Data is loaded by the class hw3Dataset.  
	1) __init__:   
		(1) Utterances time steps lengths are recorded in a list.   
		(2) Features and lengths are transferred into tensors.   
		(3) pad_sequence is used to pad list of tensors to equal length and formed a big tensor.  
	2) __getitem__:   
		return 4 elements: features, features lengths, labels, labels lengths  

## Hyperparameters
(1) Batch Size = 64 for training data loader.    
    Batch size = 64 for validation and test data loader.   
(2) optimizer:   
Adam optimizer, weight decay = 1e-5  
(3) scheduler:  
	- `MultiplicativeLR` scheduler is used. Factor = 0.85  
	- `CosineAnnealingLR` scheduler is used, eta_min = 1e-7   

(4) num_workers = 4   
(5) beam search width: 100 for final submission. 20 for validation.  

## Training Process
(1) 10 epochs with Adam optimizer and MultiplicativeLR, initial learning rate = 1e-3.  
(2) 5 epochs with Adam optimizer and CosineAnnealing scheduler, initial learning rate = 1e-4.  
(2) 10 epochs with Adam optimizer and MultiplicativeLR, initial learning rate 1e-4.  
(4) 5 epochs with Adam optimizer Cosine Annealing scheduler, initial learning rate = 1e-5  

## Network Architecture
The network architecture is defined in `class Model`.  
It is a wider and deeper version of baseline model.   
- There are 5 stacked LSTM layers, each of 512 hidden units.    
- followed by two linear layers: [1024, 1024, 42]. Between linear layers, LeakyReLU is used as the activation function followed by Dropout(0.1).  

`CTCLoss` is used as the criterion. `CTCBeamDecoder` is used for decoding.  
