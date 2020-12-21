import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader


def parse_verify_data(verify_file):
    """load verification pairs as list """
    with open(verify_file, 'r') as f:
        lines = f.readlines()
        lines = [l.strip("\n").split(" ") for l in lines]

    print("# of verify pairs:", len(lines))
    return np.array(lines)


from tqdm import tqdm

class VerifyDataset(Dataset):
    def __init__(self, root_path, pairs):
        self.root_path = root_path
        assert (pairs.shape[1] == 3)

        img1_tensors = []
        img2_tensors = []

        for img1_path, img2_path, label in tqdm(pairs):
            img1 = Image.open(self.root_path + img1_path)
            img2 = Image.open(self.root_path + img2_path)
            img1 = torchvision.transforms.ToTensor()(img1)
            img2 = torchvision.transforms.ToTensor()(img2)
            img1_tensors.append(img1)
            img2_tensors.append(img2)

        self.img1 = torch.stack(img1_tensors)
        self.img2 = torch.stack(img2_tensors)
        self.labels = pairs[:, 2]

    def __len__(self):
        return len(self.img1)

    def __getitem__(self, index):
        img1 = self.img1[index]
        img2 = self.img2[index]
        label = self.labels[index]
        return img1, img2, label


class submitDataset(Dataset):
    def __init__(self, root_path, pairs):
        self.root_path = root_path
        assert (pairs.shape[1] == 2)

        img1_tensors = []
        img2_tensors = []

        for img1_path, img2_path in tqdm(pairs):
            img1 = Image.open(self.root_path + img1_path)
            img2 = Image.open(self.root_path + img2_path)
            img1 = torchvision.transforms.ToTensor()(img1)
            img2 = torchvision.transforms.ToTensor()(img2)
            img1_tensors.append(img1)
            img2_tensors.append(img2)

        self.img1 = torch.stack(img1_tensors)
        self.img2 = torch.stack(img2_tensors)

    def __len__(self):
        return len(self.img1)

    def __getitem__(self, index):
        img1 = self.img1[index]
        img2 = self.img2[index]
        return img1, img2