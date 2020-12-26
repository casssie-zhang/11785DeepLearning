import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from models import Seq2Seq
from train_test import train, validation, test
from dataloader import load_data, collate_train, collate_test, transform_letter_to_index, Speech2TextDataset, transform_index_to_letter
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR, MultiplicativeLR, CosineAnnealingLR
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

hypers = {
    'save_path':"/content/gdrive/My Drive/11685deeplearning/hw4p2/" if DEVICE == 'cuda' else './models/',
    'model_name':"baseline",
    'nepochs':25,
    'batch_size':64 if DEVICE == 'cuda' else 2,
    'encoder_hidden_dim':256,
    'decoder_hidden_dim':512,
    'embedding_dim':256,
    'data_path': "./data/",
    'init_learning_rate': 0.001,
}


LETTER_LIST = ['<pad>', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', \
               'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '-', "'", '.', '_', '+', ' ','<sos>','<eos>']

def main():
    criterion = nn.CrossEntropyLoss(reduction='none')
    print("=== Start loading data ===")
    speech_train, speech_valid, speech_test, transcript_train, transcript_valid = load_data(hypers['data_path'])
    print("=== Successfully Loaded Data ===")
    character_text_train = transform_letter_to_index(transcript_train, LETTER_LIST)
    character_text_valid = transform_letter_to_index(transcript_valid, LETTER_LIST)

    # pre calculate target sentences in dev set
    sentences_valid = transform_index_to_letter([dev_sent[1:] for dev_sent in character_text_valid])

    print("=== Data Loaders ===")
    # DataLoader
    train_dataset = Speech2TextDataset(speech_train, character_text_train)
    dev_dataset = Speech2TextDataset(speech_valid, character_text_valid)
    test_dataset = Speech2TextDataset(speech_test, None, False)
    train_loader = DataLoader(train_dataset, batch_size=hypers['batch_size'], shuffle=True, collate_fn=collate_train)
    dev_loader = DataLoader(dev_dataset, batch_size=hypers['batch_size'], shuffle=False, collate_fn=collate_train)
    test_loader = DataLoader(test_dataset, batch_size=hypers['batch_size'], shuffle=False, collate_fn=collate_test)



    min_val_loss = 1e5
    print("=== Define Model ===")
    model = Seq2Seq(input_dim=40, vocab_size=len(LETTER_LIST),
                    encoder_hidden_dim=hypers['encoder_hidden_dim'],
                    decoder_hidden_dim=hypers['decoder_hidden_dim'],
                    embedding_dim=hypers['embedding_dim'], isAttended=True)
    print(model)

    print("=== Define optimizer and scheduler ===")
    optimizer = optim.Adam(model.parameters(), lr=hypers['init_learning_rate'])
    lmbda = lambda epoch: 0.85
    scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)
    print("optimizer:", optimizer.state_dict())
    print("scheduler:", scheduler.state_dict())


    print("=== Start Training ===")
    for epoch in range(hypers['nepochs']):
        train_loss = train(model, train_loader, criterion, optimizer, epoch, save_path=hypers['save_path'], model_name=hypers['model_name'])
        train_loss.extend(train_loss)
        val_loss = validation(model, dev_loader, criterion, epoch, sentences_valid, print_samples=True)
        val_loss.extend(val_loss)
        scheduler.step()

        if np.mean(val_loss) < min_val_loss:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, hypers['save_path'] + hypers['model_name'] + "_ep{}".format(epoch+1) + "_end" + "_" + "{:.2f}".format(np.mean(val_loss)))
        min_val_loss = np.mean(val_loss)



if __name__ == '__main__':
    main()