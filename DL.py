import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
import os
from tqdm import tqdm
import wandb
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to data csv file.", type=str, default="./aug_train_preprocessed_onehot.csv")
    parser.add_argument("--ckpt_dir", help="Checkpoints directory.", type=str, default="./checkpoints/")
    parser.add_argument("--weighted_sampler", help="Use weighted sampler or not. (Yes or No)", type=str, default="No")
    parser.add_argument("--run_name", help="Name of the run for wandb", type=str, default="base")
    return parser


parser = get_parser()
args = parser.parse_args()
data_path = args.data_path
ckpt_dir = args.ckpt_dir
weighted = args.weighted_sampler
run_name = args.run_name

wandb.login()

if not os.path.exists(ckpt_dir):
    print(f"Creating checkpoints directory...")
    os.mkdir(ckpt_dir)


def train_valid_split(data_set, valid_ratio, seed):
    """Split provided training data into training set and validation set"""
    valid_set_size = int(valid_ratio * len(data_set))
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)


data = pd.read_csv(data_path)

balance_check = dict(data["target"].value_counts())
print(f"Total number of data: {data.shape[0]}")
print(f"Total number of features: {data.shape[1]}")
print(f"Number of people not looking for job change: {balance_check[0]}")
print(f"Number of people looking for a job change: {balance_check[1]}")

numpy_data = data.values
train_data, valid_data = train_valid_split(numpy_data, valid_ratio=0.2, seed=2023)
print(f"Training data: {train_data.shape[0]}")
print(f"Validation data: {valid_data.shape[0]}")

train_data_target = train_data[:, -1]
counts = np.array([len(np.where(train_data_target == t)[0]) for t in np.unique(train_data_target)])
weights = counts / sum(counts)
weights = weights[::-1]
samples_weight = np.array([weights[int(t)] for t in train_data_target])
samples_weight = torch.from_numpy(samples_weight)
weighted_sampler = WeightedRandomSampler(samples_weight.type("torch.DoubleTensor"), len(samples_weight))


class JobDataset(Dataset):
    def __init__(self, data):
        self.labels = torch.FloatTensor(data[:, -1])
        self.features = torch.FloatTensor(data[:, :-1])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        y = self.labels[idx]
        x = self.features[idx]
        return x, y


class DNN(nn.Module):
    def __init__(self, in_dim):
        super(DNN, self).__init__()
        self.in_dim = in_dim

        self.dnn = nn.Sequential(
            nn.Linear(self.in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.dnn(x)
        out = self.sigmoid(out)
        out = out.squeeze()
        return out


def gradient_norm(model):
    norm = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            if m.weight is not None and m.weight.grad is not None:
                norm.append(torch.linalg.norm(m.weight.grad))
            if m.bias is not None and m.bias.grad is not None:
                norm.append(torch.linalg.norm(m.bias.grad))

    return sum(norm) / len(norm)


device = "cuda" if torch.cuda.is_available() else "cpu"
config = {
    "input dim": data.shape[1] - 1,
    "num classes": 2,
    "epoch": 300,
    "batch size": 64,
    "learning rate": 1e-03,
    "scheduler patience": 3,
    "scheduler min lr": 1e-05,
    "save_path": ckpt_dir,
}

train_set = JobDataset(train_data)
val_set = JobDataset(valid_data)

if weighted == "No":
    train_loader = DataLoader(train_set, batch_size=config["batch size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config["batch size"], shuffle=False)
else:
    train_loader = DataLoader(train_set, batch_size=config["batch size"], shuffle=False, sampler=weighted_sampler)
    val_loader = DataLoader(val_set, batch_size=config["batch size"], shuffle=False)

model = DNN(config["input dim"])
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=config["learning rate"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=config["scheduler patience"], min_lr=config["scheduler min lr"], verbose=True)
n_epochs = config["epoch"]
criterion = nn.BCELoss()

wandb.init(project="Job Change Prediction", config=config, name=run_name)

best_val_acc = 0
for epoch in range(1, n_epochs + 1):
    model.train()
    train_loss_record = []
    train_acc = []
    grad_norm = []
    print(f"Epoch {epoch}...")
    for i, (instances, labels) in enumerate(train_loader):
        instances, labels = instances.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(instances)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        train_loss_record.append(loss.detach().item())
        acc = (preds.round() == labels.detach()).float().mean()
        train_acc.append(acc)
        grad_norm.append(gradient_norm(model))
    print(f"Training epoch {epoch}: Accuracy = {sum(train_acc) / len(train_acc)}, Loss = {sum(train_loss_record) / len(train_loss_record)}")
    wandb.log({"Training Accuracy": sum(train_acc) / len(train_acc), "Training Loss": sum(train_loss_record) / len(train_loss_record), "Gradient Norm": sum(grad_norm) / len(grad_norm)})

    model.eval()
    val_loss_record = []
    val_acc = []

    for i, (instances, labels) in enumerate(val_loader):
        instances, labels = instances.to(device), labels.to(device)
        with torch.no_grad():
            preds = model(instances)
            loss = criterion(preds, labels)
            val_loss_record.append(loss.detach().item())
            acc = (preds.round() == labels.detach()).float().mean()
            val_acc.append(acc)
    mean_acc = sum(val_acc) / len(val_acc)
    print(f"Validation epoch {epoch}: Accuracy = {mean_acc}, Loss = {sum(val_loss_record) / len(val_loss_record)}")
    wandb.log({"Validation Accuracy": mean_acc, "Validation Loss": sum(val_loss_record) / len(val_loss_record)})

    scheduler.step(mean_acc)
    if mean_acc > best_val_acc:
        torch.save(model.state_dict(), os.path.join(config["save_path"], "best.pth"))  # Save your best model
        print("Saving model with accuracy {:.3f}...".format(mean_acc))
        best_val_acc = mean_acc

wandb.finish()
