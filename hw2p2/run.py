from resnet import MyResNet34, init_weights
# from baseline import baselineModel
from train import train
from wider_baseline import baselineModel as widerBaseline

from dataloader import VerifyDataset, parse_verify_data
import torch
import torch.nn as nn
import torchvision

batch_size = 256
num_workers = 4

num_feats = 3 # input channel
feat_dim = 512 #embedding size
learningRate = 0.15 # initial learning rate
weightDecay = 5e-5
num_classes = 4000
model_name = 'wider_baseline'
save_path = "/content/gdrive/My Drive/11685deeplearning/hw2p2/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""data loader"""
# data augmentation
from torchvision import transforms
TRANSFORM_IMG = transforms.Compose([
    transforms.RandomHorizontalFlip(), #horizontal flip
    transforms.ToTensor(),
    ])

root_image_path = "/content/datasets/cmu11785/20fall-hw2p2/classification_data/"
verify_root_path = "/content/datasets/cmu11785/20fall-hw2p2/"
verify_file = "/content/datasets/cmu11785/20fall-hw2p2/verification_pairs_val.txt"
verify_pairs = parse_verify_data(verify_file)


train_dataset = torchvision.datasets.ImageFolder(root=root_image_path + "train_data",
                                                 transform=TRANSFORM_IMG)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)

dev_dataset = torchvision.datasets.ImageFolder(root=root_image_path + "val_data",
                                               transform=torchvision.transforms.ToTensor())
dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

verify_dataset = VerifyDataset(verify_root_path, verify_pairs)
verify_dataloader = torch.utils.data.DataLoader(verify_dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

"""network"""

hidden_sizes = [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3
network = widerBaseline(input_channel=3, output_size=4000)
# network = baselineModel(input_channel=3, output_size=4000)
network.apply(init_weights)
network.train()
network.to(device)

optimizer = torch.optim.SGD(network.parameters(),
                            lr=learningRate,
                            weight_decay=weightDecay, momentum=0.9)

from torch.optim.lr_scheduler import MultiplicativeLR
lmbda = lambda epoch: 0.85
scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)

criterion = nn.CrossEntropyLoss()


# hidden_sizes = [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3
# num_classes = len(train_dataset.classes)
numEpochs = 12
train(model=network, data_loader=train_dataloader, test_loader=dev_dataloader, verify_loader=verify_dataloader,
      numEpochs=numEpochs, device=device, optimizer=optimizer, scheduler=scheduler, criterion=criterion,
      if_save=True, model_name = model_name, save_path = save_path)

numEpochs = 10
optimizer = torch.optim.Adam(network.parameters(), lr=1e-4, weight_decay=weightDecay)
scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)
train(model=network, data_loader=train_dataloader, test_loader=dev_dataloader, verify_loader=verify_dataloader,
      numEpochs=numEpochs, device=device, optimizer=optimizer, scheduler=scheduler, criterion=criterion,
      if_save=True, model_name = model_name, save_path = save_path)