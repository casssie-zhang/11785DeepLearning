import time
import torch
### Add Your Other Necessary Imports Here! ###
import numpy as np
import pandas as pd
from dataloader import transform_index_to_letter
from models import Seq2Seq
from Levenshtein import distance as levenshtein_distance


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def lev_edit_distance(predicts, targets):
    """calculate edit distance"""
    assert(len(predicts) == len(targets))
    distance_list = []
    for pred, target in zip(predicts, targets):
        distance = levenshtein_distance(pred, target)
    distance_list.append(distance)

    return np.mean(distance_list)



def train(model:Seq2Seq, train_loader, criterion, optimizer, epoch, save_path, model_name):
    """train one epoch!"""
    model.train()
    model.to(DEVICE)
    start = time.time()
    loss_history = []
        # 1) Iterate through your loader
    for idx, (padded_speech, padded_text, speech_lens, text_lens) in enumerate(train_loader):
        # 2) Use torch.autograd.set_detect_anomaly(True) to get notices about gradient explosion
        model.train()

        # 3) Set the inputs to the device.
        padded_speech, padded_text, speech_lens, text_lens = \
            padded_speech.to(DEVICE), padded_text.to(DEVICE), speech_lens.to(DEVICE), text_lens.to(DEVICE)

        optimizer.zero_grad()
        torch.cuda.empty_cache()

        # 4) Pass your inputs, and length of speech into the model.
        predictions = model.forward(padded_speech, speech_lens, padded_text, teacher_forcing_ratio=0.1)

        del padded_speech, speech_lens

        # 5) Generate a mask based on the lengths of the text to create a masked loss.
        # 5.1) Ensure the mask is on the device and is the correct shape.
        # padded text start from zero, so use < rather than <=
        loss_mask = torch.arange(padded_text.size(1)).unsqueeze(0) < text_lens.cpu().unsqueeze(1)
        loss_mask = loss_mask.to(DEVICE)


        # 7) Use the criterion to get the loss.
        loss = criterion(predictions.permute(0,2,1), padded_text)
        # 8) Use the mask to calculate a masked loss.
        masked_loss = (loss * loss_mask).sum() / loss_mask.sum()

        del padded_text, text_lens, predictions
        torch.cuda.empty_cache()

        # 9) Run the backward pass on the masked loss.
        masked_loss.backward()

        # 10) Use torch.nn.utils.clip_grad_norm(model.parameters(), 2)
        torch.nn.utils.clip_grad_norm_(model.parameters(),2)

        # 11) Take a step with your optimizer
        optimizer.step()

        with torch.no_grad():
            # 12) Normalize the masked loss # perplexity
            perplexity = torch.exp(masked_loss).item()
            loss_history.append(perplexity)
            # 13) Optionally print the training loss after every N batches
            if idx % 50 == 49:
                took = time.time()-start
                start = time.time()
                print("Epoch = {}\tbatch = {}\tTrain Perplexity: {:.2f}\tMasked Loss: {:.2f}\tTime: {}sec"\
                      .format(epoch+1, idx+1, perplexity, masked_loss.item(), int(took)))

            # if idx % 200 == 199: # save checkpoint per 200 batches: this is because cuda error
            #     torch.save({
            #         'model_state_dict': model.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #     }, save_path + model_name + "_ep{}".format(epoch+1) + "_b{}".format(idx+1) + "_" + "{:.2f}".format(perplexity))
    return loss_history


def validation(model, dev_loader, criterion, epoch, target_text, print_samples=False):
    model.eval()
    model.to(DEVICE)
    val_loss = []
    dev_predict_text = [] # predict result
    # dev_predict_text_random = []
    with torch.no_grad():

        for idx, (padded_speech, padded_text, speech_lens, text_lens) in enumerate(dev_loader):

            padded_speech, padded_text, speech_lens, text_lens = \
                padded_speech.to(DEVICE), padded_text.to(DEVICE), speech_lens.to(DEVICE), text_lens.to(DEVICE)

            predictions = model.forward(padded_speech, speech_lens, padded_text, isTrain=False)

            del padded_speech, speech_lens
            torch.cuda.empty_cache()

            # 1. calculate loss
            loss_mask = torch.arange(padded_text.size(1)).unsqueeze(0) < text_lens.cpu().unsqueeze(1)
            loss_mask = loss_mask.to(DEVICE)

            max_len = padded_text.size(1)
            loss = criterion(predictions.permute(0,2,1)[:,:,:max_len], padded_text)
            masked_loss = (loss * loss_mask).sum() / loss_mask.sum()

            del padded_text, text_lens
            torch.cuda.empty_cache()

            perplexity = masked_loss.exp().item()
            val_loss.append(perplexity)

            # 2. decoding using greedy
            predictions_arr = predictions.argmax(-1).detach().cpu().numpy()
            prediction_text = transform_index_to_letter(predictions_arr)
            dev_predict_text.extend(prediction_text)

            # decoding using random search
            # raw_sentence_arr = predicted_sent.detach().cpu().numpy()
            # raw_sentence_text = transform_index_to_letter(raw_sentence_arr)
            # dev_predict_text_random.extend(raw_sentence_text)


        distance = lev_edit_distance(dev_predict_text, target_text)
        # distance_random = lev_edit_distance(dev_predict_text_random, target_text)
        # print samples
        if print_samples:
            print("== print 5 samples ==")
            for pred, target in zip(dev_predict_text[:5], target_text[:]):
                print("prediction:",pred)
                print("targets:", target)
            print("=====================")


        mean_val_loss = np.mean(val_loss)
        print("Epoch = {}\tVal Perplexity: {:.4f}, Edit Distance: {:.4f}".format(epoch, mean_val_loss, distance))

    return val_loss

def test(model:Seq2Seq, test_loader, criterion, csv_name="submission.csv"):
    ### Write your test code here! ###
    model.eval()
    model.to(DEVICE)
    test_predict_text = [] # predict result
    dev_predict_text_random = []
    print("=== start testing ===")
    with torch.no_grad():
        for idx, (padded_speech, speech_lens) in enumerate(test_loader):
            padded_speech, speech_lens = \
                padded_speech.to(DEVICE), speech_lens.to(DEVICE)

            # loss = criterion(predictions.permute(0, 2, 1)[:, :, :max_len], padded_text)
            # masked_loss = (loss * loss_mask).sum() / loss_mask.sum()

            # 4) Pass your inputs, and length of speech into the model.
            for i in range(100):
                predictions, predicted_sent = model.forward(padded_speech, speech_lens, isTrain=False)
                # 2. decoding using greedy
                predictions_arr = predictions.argmax(-1).detach().cpu().numpy()

            prediction_text = transform_index_to_letter(predictions_arr)
            test_predict_text.extend(prediction_text)

            raw_sentence_arr = predicted_sent.detach().cpu().numpy()
            raw_sentence_text = transform_index_to_letter(raw_sentence_arr)
            dev_predict_text_random.extend(raw_sentence_text)

    print("save")
    submission = pd.DataFrame(enumerate(test_predict_text), columns=['id', 'label'])
    submission.to_csv(csv_name, index=False)

    # submission = pd.DataFrame(enumerate(test_predict_text), columns=['id', 'label'])
    # submission.to_csv("random_submission.csv")