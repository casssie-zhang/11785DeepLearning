import time
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F




def train(model, data_loader, test_loader, verify_loader, numEpochs, device, optimizer, scheduler,
          criterion, if_save = False, model_name = None, save_path = "./model/"):
    model.train()

    start = time.time()
    for epoch in range(numEpochs):
        avg_loss = 0.0
        for batch_num, (feats, labels) in enumerate(data_loader):
            model.train()
            feats, labels = feats.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(feats)[1]  # label output

            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

            if batch_num % 50 == 49:
                print('Epoch: {}/Batch: {}\tAvg-Loss: {:.4f}\t time: {}sec'.format(epoch + 1, batch_num + 1,
                                                                                   avg_loss / 50,
                                                                                   int(time.time() - start)))
                start = time.time()
                # scheduler.step(avg_loss)
                avg_loss = 0.0

            torch.cuda.empty_cache()
            del feats
            del labels
            del loss

        # if task == 'Classification':
        val_loss, val_acc = test_classify(model, test_loader, criterion, device)
        train_loss, train_acc = test_classify(model, data_loader, criterion, device)

        auc = test_verify(model, verify_loader, device)

        print('Train Loss: {:.4f}\tTrain Accuracy: {:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}\tAUC: {:.3f}'.
              format(train_loss, train_acc, val_loss, val_acc, auc))

        scheduler.step()
        # scheduler.step(val_loss)
        print("learning rate: ", scheduler.get_last_lr())

        if if_save and model_name:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, save_path + model_name + "_" + str(epoch) + "_" + "{:.2f}".format(auc))


def test_classify(model, test_loader, criterion, device):
    model.eval()
    test_loss = []
    accuracy = 0
    total = 0

    for batch_num, (feats, labels) in enumerate(test_loader):
        feats, labels = feats.to(device), labels.to(device)
        outputs = model(feats)[1]

        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)

        loss = criterion(outputs, labels.long())

        accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        test_loss.extend([loss.item()] * feats.size()[0])
        del feats
        del labels

    model.train()
    return np.mean(test_loss), accuracy / total


def test_verify(model, verify_dataloader, device):
    model.eval()

    sims_list = []
    labels_list = []

    for batch_num, (img1, img2, labels) in enumerate(verify_dataloader):
        img1, img2 = img1.to(device), img2.to(device)

        img1_feats = model(img1)[2]
        img2_feats = model(img2)[2]

        # compute similarity
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        sims = cos(img1_feats, img2_feats)

        sims_list.extend(sims.cpu().detach().numpy())
        labels_list.extend(labels)

        del img1_feats
        del img2_feats
        torch.cuda.empty_cache()

    roc_score = roc_auc_score(labels_list, sims_list)

    model.train()

    return roc_score
