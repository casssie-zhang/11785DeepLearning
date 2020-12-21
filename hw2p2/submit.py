from dataloader import parse_verify_data, submitDataset
from wider_baseline import baselineModel as widerBaseline
import torch
import pandas as pd
from tqdm import tqdm

verify_file = "datasets/cmu11785/20fall-hw2p2/verification_pairs_test.txt"
verify_root_path = "datasets/cmu11785/20fall-hw2p2/"
batch_size = 256
num_workers = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

verify_pairs = parse_verify_data(verify_file)
submit_dataset = submitDataset(verify_root_path, verify_pairs)
submit_dataloader = torch.utils.data.DataLoader(submit_dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

def submit_csv(model, pairs, verify_dataloader, device, submit_name = "submission.csv"):
    model.eval()

    sims_list = []
    labels_list = []

    for batch_num, (img1, img2) in tqdm(enumerate(verify_dataloader)):
        img1, img2 = img1.to(device), img2.to(device)

        img1_feats = model(img1)[2]
        img2_feats = model(img2)[2]

        # compute similarity
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        sims = cos(img1_feats, img2_feats)

        sims_list.extend(sims.cpu().detach().numpy())
        # labels_list.extend(labels)

        del img1_feats
        del img2_feats
        torch.cuda.empty_cache()
    model.train()

    data_list = []
    for (img1, img2), sim in zip(pairs, sims_list):
        data_list.append([" ".join([img1, img2]), sim])

    df = pd.DataFrame(data_list, columns = ['Id', 'Category'])
    df.to_csv(submit_name, index=False)

model_file = "model/follow_wider_baseline_4_0.91"
from baseline import baselineModel
model = baselineModel(3, 4000)
temp = torch.load(model_file)
model.load_state_dict(temp['model_state_dict'])
model.to(device)
submit_csv(model, verify_pairs, submit_dataloader, device, submit_name="baseline_submission_3.csv")