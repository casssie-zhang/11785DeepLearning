import numpy as np
import torch
from torch.utils.data import Dataset 
from torch.nn.utils.rnn import *

LETTER_LIST = ['<pad>', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', \
               'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '-', "'", '.', '_', '+', ' ','<sos>','<eos>']


'''
Loading all the numpy files containing the utterance information and text information
'''
def load_data(data_path):
    speech_train = np.load(data_path+'train.npy', allow_pickle=True, encoding='bytes')
    speech_valid = np.load(data_path+'dev.npy', allow_pickle=True, encoding='bytes')
    speech_test = np.load(data_path+'test.npy', allow_pickle=True, encoding='bytes')

    transcript_train = np.load(data_path+'train_transcripts.npy', allow_pickle=True,encoding='bytes')
    transcript_valid = np.load(data_path+'dev_transcripts.npy', allow_pickle=True,encoding='bytes')
    transcript_train = np.array([np.array([word.decode('utf-8') for word in sentence]) for sentence in transcript_train])
    transcript_valid = np.array([np.array([word.decode('utf-8') for word in sentence]) for sentence in transcript_valid])


    return speech_train, speech_valid, speech_test, transcript_train, transcript_valid


'''
Transforms alphabetical input to numerical input, replace each letter by its corresponding 
index from letter_list
'''
def transform_letter_to_index(transcript, letter_list=LETTER_LIST):
    '''
    add <sos>, blank, <eos>
    :param transcript :(N, ) Transcripts are the text input
    :param letter_list: Letter list defined above
    :return letter_to_index_list: Returns a list for all the transcript sentence to index
    '''
    letter2index, _ = create_dictionaries(letter_list)
    letter_to_index_list = [[letter2index['<sos>']] +
                            [letter2index[letter] for letter in " ".join(sentence)] +
                            [letter2index['<eos>']]
                            for sentence in transcript]
    return letter_to_index_list

def transform_index_to_letter(indexes, letter_list=LETTER_LIST, eos_idx = [34,0]):
    _, index2letter = create_dictionaries(letter_list)
    sentence_list = []
    for sentence in indexes:
        sentence_letters = []
        for letter in sentence:
            if letter in eos_idx:
                break
            else:
                sentence_letters.append(index2letter[letter])
        sentence = ''.join(sentence_letters)
        sentence_list.append(sentence)
    return sentence_list



'''
Optional, create dictionaries for letter2index and index2letter transformations
'''
def create_dictionaries(letter_list):
    index2letter = dict(enumerate(letter_list))
    letter2index = {v:k for k, v in index2letter.items()}
    return letter2index, index2letter


class Speech2TextDataset(Dataset):
    '''
    Dataset class for the speech to text data, this may need some tweaking in the
    getitem method as your implementation in the collate function may be different from
    ours. 
    '''
    def __init__(self, speech, text=None, isTrain=True):
        self.speech = speech
        self.isTrain = isTrain
        if (text is not None):
            self.text = text

    def __len__(self):
        return self.speech.shape[0]

    def __getitem__(self, index):
        if (self.isTrain == True):
            return torch.tensor(self.speech[index].astype(np.float32)), torch.tensor(self.text[index])
        else:
            return torch.tensor(self.speech[index].astype(np.float32))


def collate_train(batch_data):
    ### Return the padded speech and text data, and the length of utterance and transcript ###

    speech = [i[0] for i in batch_data]
    text = [i[1] for i in batch_data]

    speech_lens = torch.LongTensor([len(i) for i in speech])
    text_lens = torch.LongTensor([len(i) for i in text]) - 1

    padded_speech = pad_sequence(speech, batch_first=True)
    padded_text = pad_sequence(text, batch_first=True)[:,1:] # skip <sos> -> raw text+eos

    return padded_speech, padded_text, speech_lens, text_lens


def collate_test(batch_data):
    ### Return padded speech and length of utterance ###
    speech = batch_data
    speech_lens = torch.LongTensor([len(i) for i in speech])
    padded_speech = pad_sequence(speech, batch_first = True)

    return padded_speech, speech_lens
