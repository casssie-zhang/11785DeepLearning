import torch
import torch.nn as nn
import numpy as np
import torch.nn.utils as utils
from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import seaborn as sns

from torch.distributions.categorical import Categorical

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
teacher_forcing_ratio = 0.1

def pass_to_lstm(lstm, x, lens, batch_first = True):
    """
    a wrapper for lstm with packing and unpacking sequence
    """
    rnn_inp = pack_padded_sequence(x, lengths=lens.cpu(), batch_first=batch_first, enforce_sorted=False)
    del x
    outputs, _ = lstm(rnn_inp)
    del rnn_inp
    outputs, out_lens = pad_packed_sequence(outputs, batch_first=batch_first)
    return outputs, out_lens


class Attention(nn.Module):
    '''
    Attention is calculated using key, value and query from Encoder and decoder.
    Below are the set of operations you need to perform for computing attention:
        energy = bmm(key, query)
        attention = softmax(energy)
        context = bmm(attention, value)
    '''
    def __init__(self):
        super(Attention, self).__init__()
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, lens):
        '''
        :param query :(batch_size, hidden_size) Query is the output of LSTMCell from Decoder
        :param keys: (batch_size, max_len, encoder_size) Key Projection from Encoder
        :param values: (batch_size, max_len, encoder_size) Value Projection from Encoder
        :return context: (batch_size, encoder_size) Attended Context
        :return attention_mask: (batch_size, max_len) Attention mask that can be plotted
        '''
        # input shape (batch_size, max_len, encoder_size), (batch_size, hidden_size, 1)
        # output shape (batch_size, max_len, 1)
        energy = torch.bmm(key, query.unsqueeze(2)).squeeze(2)


        # Create an (batch_size, max_len) boolean mask for all padding positions
        # Make use of broadcasting: (1, max_len), (batch_size, 1) -> (batch_size, max_len)
        attention_mask = torch.arange(value.size(1)).unsqueeze(0) >= lens.unsqueeze(1) #TODO: <= or >=
        attention_mask = attention_mask.to(DEVICE)
        energy.masked_fill_(attention_mask, -1e9) # TODO
        attention = nn.functional.softmax(energy, dim=1)
        # shape = (batch_size, 1, encoder_size)
        context = torch.bmm(attention.unsqueeze(1), value).squeeze(1)

        return context, attention



class pBLSTM(nn.Module):
    '''
    Pyramidal BiLSTM
    The length of utterance (speech input) can be hundereds to thousands of frames long.
    The Paper reports that a direct LSTM implementation as Encoder resulted in slow convergence,
    and inferior results even after extensive training.
    The major reason is inability of AttendAndSpell operation to extract relevant information
    from a large number of input steps.
    '''
    def __init__(self, input_dim, hidden_dim, dropout_rate = 0.0):
        super(pBLSTM, self).__init__()
        self.blstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True,
                             dropout=dropout_rate,batch_first=True)

    def forward(self, x, lens):
        '''
        :param x :(N, T) input to the pBLSTM
        :return output: (N, T, H) encoded sequence from pyramidal Bi-LSTM 
        '''
        assert (type(x) == torch.Tensor)
        batch_size, timestep, feature_dim = x.shape
        # x.shape = B, T/2, dim*2, chop data if needed.
        if timestep % 2 == 1:
            x = x[:, :-1, :] # chop

        input_x = x.contiguous().view(batch_size,int(timestep/2),feature_dim*2)
        del x
        output, output_lens = pass_to_lstm(self.blstm, input_x, lens//2)
        # output, hidden = self.blstm(input_x)
        # print("after pblstm:",output.shape)

        return output, output_lens




class Encoder(nn.Module):
    '''
    Encoder takes the utterances as inputs and returns the key and value.
    Key and value are nothing but simple projections of the output from pBLSTM network.
    '''
    def __init__(self, input_dim, hidden_dim, value_size=128,key_size=128):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=1, bidirectional=True, batch_first=True)
        # output.shape =(B, T, hidden)
        
        ### Add code to define the blocks of pBLSTMs! ###
        self.pBLSTM_1 = pBLSTM(hidden_dim * 2 * 2, hidden_dim)
        self.pBLSTM_2 = pBLSTM(hidden_dim * 2 * 2, hidden_dim)
        self.pBLSTM_3 = pBLSTM(hidden_dim * 2 * 2, hidden_dim)

        self.key_network = nn.Linear(hidden_dim*2, value_size)
        self.value_network = nn.Linear(hidden_dim*2, key_size)

        # self.listener_layer = listener_layer

    def forward(self, x, lens):
        # print("original:", x.shape)
        outputs = pass_to_lstm(self.lstm, x, lens)[0]

        outputs, out_lens = self.pBLSTM_1(outputs, lens)
        outputs, out_lens = self.pBLSTM_2(outputs, out_lens)
        outputs, out_lens = self.pBLSTM_3(outputs, out_lens)

        linear_input = outputs
        keys = self.key_network(linear_input)
        value = self.value_network(linear_input)

        return keys, value, out_lens



class Decoder(nn.Module):
    '''
    As mentioned in a previous recitation, each forward call of decoder deals with just one time step, 
    thus we use LSTMCell instead of LSLTM here.
    The output from the second LSTMCell can be used as query here for attention module.
    In place of value that we get from the attention, this can be replace by context we get from the attention.
    Methods like Gumble noise and teacher forcing can also be incorporated for improving the performance.
    '''
    def __init__(self, vocab_size, hidden_dim, embedding_dim, value_size=128, key_size=128, isAttended=False,
                 dropout_p = 0.1, tie_weights=True):
        super(Decoder, self).__init__()
        """
        vocab_size: size of dictionary
        hidden_dim: decoder dim 
        """
        # hidden dim is embedding dim -- input speech dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm1 = nn.LSTMCell(input_size=embedding_dim + value_size, hidden_size=hidden_dim)
        self.lstm2 = nn.LSTMCell(input_size=hidden_dim, hidden_size=key_size)

        # dropout=0.1
        # dropouth=0.1
        # dropouti=0.1
        # dropoute=0.1
        #
        # self.lockdrop = LockedDropout()
        # self.idrop = nn.Dropout(dropouti)
        # self.hdrop = nn.Dropout(dropouth)
        # self.drop = nn.Dropout(dropout)
        #
        # self.dropout = dropout
        # self.dropouti = dropouti
        # self.dropouth = dropouth
        # self.dropoute = dropoute


        self.isAttended = isAttended
        if (isAttended == True):
            self.attention = Attention()

        self.character_prob = nn.Linear(key_size + value_size, vocab_size)
        if tie_weights:
            self.character_prob.weight = self.embedding.weight
        
        self.attention_plot = []
        self.batch_cnt = 0

    def init_attention(self):
        assert (self.isAttended == True)
        self.attention = Attention()

    def forward_step(self, char_embed, context, hidden_states, key, values, lens):
        """
        perform one single timestep forward, make predictions
        returns:
            prediction: character distribution
            context:

        """
        inp = torch.cat([char_embed, context], dim=1)
        hidden_states[0] = self.lstm1(inp, hidden_states[0])
        inp_2 = hidden_states[0][0]
        hidden_states[1] = self.lstm2(inp_2, hidden_states[1])

        ### Compute attention from the output of the second LSTM Cell ###
        output = hidden_states[1][0]
        if self.isAttended:
            context, attention_mask = self.attention(output, key, values, lens)
            self.attention_plot.append(attention_mask.unsqueeze(2)) # attention_mask = (batch_size, input_max_len, 1)
        else:
            context = torch.zeros_like(context)
        prediction = self.character_prob(torch.cat([output, context], dim=1))

        return prediction, context, hidden_states


    def forward(self, key, values, lens, text=None, isTrain=True, teacher_forcing_ratio=0.1, isGumbel=False, isRandom=False, plot_attention=False):
        '''
        :param key :(T, N, key_size) Output of the Encoder Key projection layer
        :param values: (T, N, value_size) Output of the Encoder Value projection layer
        :param text: (N, text_len) Batch input of text with text_length
        :param isTrain: Train or eval mode
        :return predictions: Returns the character prediction probability
        '''
        batch_size = key.shape[0]
        # TODO: train, use ground truth or teacher forcing ratio
        self.batch_cnt += 1

        if (isTrain == True):
            max_len = text.shape[1]
            embeddings = self.embedding(text)
        else: # TODO: test: generate tokens
            max_len = 600

        predictions = [] # prediction probabilities
        hidden_states = [None, None]

        # initialize with <sos>
        prediction = (torch.ones(batch_size, 1)*33).to(DEVICE)
        # print("max_len:", max_len)
        encoder_size = values.size(2)
        # context = values[:,0,:]
        context = torch.zeros((batch_size, encoder_size)).to(DEVICE) # initial context with zeros

        for i in range(max_len):
            # * Implement Gumble noise and teacher forcing techniques 
            # * When attention is True, replace values[i,:,:] with the context you get from attention.
            # * If you haven't implemented attention yet, then you may want to check the index and break 
            #   out of the loop so you do not get index out of range errors.
            # TODO: decide timestep input
            if (isTrain):
                use_teacher_forcing = True if random.random() > teacher_forcing_ratio else False
                if use_teacher_forcing: # use ground truth
                    char_embed = self.embedding(prediction.squeeze(1).long()) if i==0 else embeddings[:, i-1, :]
                else:
                    if isRandom:
                        prediction_prob = prediction.softmax(1)
                        pred_chars = Categorical(prediction_prob).sample()
                        char_embed = self.embedding(pred_chars)
                    else:
                        prediction_prob = prediction.softmax(1)
                        char_embed = self.embedding(prediction_prob.argmax(dim=-1))  # greedy

                    # char_embed = embeddings[:, i, :]
            else: # TODO: generate from distribution
                # random sample to generate
                # prediction_prob = prediction.softmax(1)
                if i == 0:
                    char_embed = self.embedding(prediction.squeeze(1).long())
                else:
                    if isGumbel:
                        prediction_prob = torch.nn.functional.gumbel_softmax(prediction)
                    else:
                        prediction_prob = torch.nn.functional.softmax(prediction)

                    if isRandom:
                        char = Categorical(prediction_prob).sample()
                        char_embed = self.embedding(char)
                    else: # greedy
                        char = prediction_prob.argmax(dim=-1)
                        char_embed = self.embedding(char)

                    # predicted_chars.append(char.unsqueeze(1))

                # pred_chars = Categorical(prediction_prob).sample()
                # char_embed = self.embedding(pred_chars)

            prediction, context, hidden_states = \
                self.forward_step(char_embed, context, hidden_states, key, values, lens)



            predictions.append(prediction.unsqueeze(1))

        # if plot_attention and len(self.attention_plot) > 0:
        #     attentions_all = torch.cat(self.attention_plot, dim=2)[0] # plot the first
        #     sns_plot = sns.heatmap(attentions_all.cpu().detach().numpy())
        #     print("save_fig")
        #     sns_plot.get_figure().savefig("attention_vis_{}.png".format(self.batch_cnt))


        # if len(predicted_chars) > 0:
        #     result = torch.cat(predicted_chars, dim=1)
        # else:
        #     result = None
        return torch.cat(predictions, dim=1)


class Seq2Seq(nn.Module):
    '''
    We train an end-to-end sequence to sequence model comprising of Encoder and Decoder.
    This is simply a wrapper "model" for your encoder and decoder.
    '''
    def __init__(self, input_dim, vocab_size, encoder_hidden_dim, decoder_hidden_dim, embedding_dim, value_size=128, key_size=128, isAttended=False):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_dim, encoder_hidden_dim)
        self.decoder = Decoder(vocab_size, hidden_dim=decoder_hidden_dim, embedding_dim=embedding_dim, isAttended=isAttended)
        self.isAttended = isAttended

    def forward(self, speech_input, speech_len, text_input=None, teacher_forcing_ratio=0.2, isTrain=True, isGumbel=True, plot_attention=False):
        key, value, out_lens = self.encoder(speech_input, speech_len)

        if (isTrain == True):
            predictions = self.decoder.forward(key, value, out_lens, text_input, teacher_forcing_ratio=teacher_forcing_ratio, plot_attention=plot_attention, isGumbel=False, isRandom=True)
        else:
            predictions = self.decoder.forward(key, value, out_lens, text=None, isTrain=False, plot_attention=plot_attention, isGumbel=isGumbel, isRandom=False)
        return predictions

    def reset_decoder(self):
        self.decoder.attention_plot = []

    def set_attention(self):
        """
        train without attention
        """
        self.isAttended = True
        self.decoder.isAttended = True
        self.decoder.init_attention()
