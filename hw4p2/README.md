# HW4P2 - Attention-based End-to-End Speech-to-Text Deep Neural Network - Writeup



1. Steps to run the model  
```
# save data in data folder
mkdir data
mkdir models 
python main.py
```

2. DataLoader:  
Data is loaded by the class Speech2TextDataset.  
Transcripts processing: 
	- add blank,<sos>, <eos>  
	- transform letters to index  
Speech:  
	- transfer to FloatTensor  
	- record lengths  
	- pad  

3. Hyperparameters   
(1) Batch Size = 64 for training data loader.     
    Batch size = 64 for validation and test data loader.   
(2) optimizer:    
Adam optimizer, weight decay = 1e-5
(3) scheduler:    
	- `MultiplicativeLR` scheduler is used. Factor = 0.85   
	- `CosineAnnealingLR` scheduler is used, eta_min = 1e-7  

4. Training Process
(1) 25 epochs with Adam optimizer and MultiplicativeLR, initial learning rate = 1e-3.  
(2) increase teacher forcing ratio = 0.2, change to CosineAnnealingLR, train 10 epochs  
(3) increase teacher forcing ratio = 0.3  

5. Network Architecture
A character-based model based on <cite>[Listen, Attend And Spell][1]</cite>

Listener - Encoder:   
	- 3 layers Pyramidal Bi-LSTM Network
	- encoder_hidden_dim = 256

Speller - Decoder:   
	- 2 layers LSTM
	- decoder_hidden_dim=512
	- embedding_dim=256

Attention:   
	- key_size=128
	- value_size=128

Teacher Forcing Ratio:   
	- from 0.1 to 0.3
	- if not teacher forcing, use greedy search to generate words as input for next time step.

Decoding: 
Gumbel Softmax and greedy search is used during decoding.

[1] https://arxiv.org/abs/1508.01211