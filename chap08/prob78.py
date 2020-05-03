
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

prob_num = 78
checkpoint = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device :", device)

class NeuralNet(nn.Module):
    def __init__(self, vector_size, output_size=4):
        super(NeuralNet, self).__init__()
        self.linear = nn.Linear(vector_size, output_size, bias=False)
        self.out = nn.Softmax(dim=1)
    
    def forward(self, x):

        h = self.linear(x)
        out = self.out(h)

        return out

def loss_fn(pred, label):
    loss = nn.CrossEntropyLoss()(pred, label)
    return loss

X_train = np.array(list(joblib.load('./data/X_train.joblib')))
y_train = joblib.load('./data/y_train.joblib')
X_valid = np.array(list(joblib.load('./data/X_valid.joblib')))
y_valid = joblib.load('./data/y_valid.joblib')

X_train = torch.from_numpy(X_train.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.int64))
X_valid = torch.from_numpy(X_valid.astype(np.float32))
y_valid = torch.from_numpy(y_valid.astype(np.int64))

X = X_train
y = y_train

X_valid = X_valid.to(device)
y_valid = y_valid.to(device)

train_dataset = TensorDataset(X, y)

train_loss_all = []
train_acc_all = []
valid_loss_all = []
valid_acc_all = []

batch_list = [2, 4, 8, 16, 32]
for bs in batch_list:
    train_loss_temp = []
    train_acc_temp = []
    valid_loss_temp = []
    valid_acc_temp = []

    model = NeuralNet(X_train.shape[1], 4)
    model = model.to(device)
    optimizer = SGD(model.parameters(), lr=0.01)

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    print(f"batch size : {bs}")
    for epoch in tqdm(range(100)):

        trian_loss = 0.0
        trian_acc = 0.0

        for batch_x, batch_y in train_loader:
            
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            model = model.train()
            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y)

            loss.backward()
            optimizer.step()

            trian_loss += loss.item() / len(train_loader)

            _, y_pred = torch.max(y_pred, 1)
            trian_acc += (y_pred == batch_y).sum().item() / len(y)
        
        train_loss_temp.append(trian_loss)
        train_acc_temp.append(trian_acc)

        if epoch % checkpoint == 0:
            torch.save(model.state_dict(), f"./data/prob{prob_num}_model_ckpt_{epoch}.bin")
            torch.save(optimizer.state_dict(), f"./data/prob{prob_num}_optimizer_ckpt_{epoch}.bin")

        model = model.eval()
        with torch.no_grad():
            y_pred = model(X_valid)
            loss = loss_fn(y_pred, y_valid)

            _, y_pred = torch.max(y_pred, 1)
            acc = (y_pred == y_valid).sum().item() / len(y_valid)
            valid_loss_temp.append(loss)
            valid_acc_temp.append(acc)
    
    train_loss_all.append(train_loss_temp)
    train_acc_all.append(train_acc_temp)
    valid_loss_all.append(valid_loss_temp)
    valid_acc_all.append(valid_acc_temp)

list_epoch = list(range(100))

plt.figure(figsize=(15, 5))
plt.subplot(1,2,1)
for idx in range(len(batch_list)):
    bs = batch_list[idx]
    train_acc_idx = train_acc_all[idx]
    valid_acc_idx = valid_acc_all[idx]

    plt.plot(list_epoch, train_acc_idx, label=f"train batch{bs}")
    plt.plot(list_epoch, valid_acc_idx, label=f"validation batch{bs}")
plt.legend()
plt.xlabel("epoch", fontsize = 18)
plt.ylabel("accuracy", fontsize = 18)
plt.tick_params(labelsize=14)

plt.subplot(1,2,2)
for idx in range(len(batch_list)):
    bs = batch_list[idx]
    train_loss_idx = train_loss_all[idx]
    valid_loss_idx = valid_loss_all[idx]

    plt.plot(list_epoch, train_loss_idx, label=f"train batch{bs}")
    plt.plot(list_epoch, valid_loss_idx, label=f"validation batch{bs}")
plt.legend()
plt.xlabel("EPOCH", fontsize = 18)
plt.ylabel("loss", fontsize = 18)
plt.tick_params(labelsize=14)

plt.tight_layout()
plt.savefig(f"./data/prob{prob_num}.png")