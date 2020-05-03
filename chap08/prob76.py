import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

prob_num = 76

class NeuralNet(nn.Module):
    def __init__(self, vector_size, output_size=4):
        super(NeuralNet, self).__init__()
        self.linear = nn.Linear(vector_size, output_size, bias=False)
        self.out = nn.Softmax(dim=1)
    
    def forward(self, x, label=None):

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

model = NeuralNet(X_train.shape[1], 4)
optimizer = SGD(model.parameters(), lr=0.01)

train_loss = []
train_acc = []
valid_loss = []
valid_acc = []

checkpoint = 10

for epoch in tqdm(range(100)):
    model = model.train()
    y_pred = model(X)
    loss = loss_fn(y_pred, y)

    loss.backward()
    optimizer.step()

    _, y_pred = torch.max(y_pred, 1)
    acc = (y_pred == y).sum().item() / len(y)
    train_loss.append(loss)
    train_acc.append(acc)

    if epoch % checkpoint == 0:
        torch.save(model.state_dict(), f"./data/prob{prob_num}_model_ckpt_{epoch}.bin")
        torch.save(optimizer.state_dict(), f"./data/prob{prob_num}_optimizer_ckpt_{epoch}.bin")

    model = model.eval()
    with torch.no_grad():
        y_pred = model(X_valid)
        loss = loss_fn(y_pred, y_valid)

        _, y_pred = torch.max(y_pred, 1)
        acc = (y_pred == y_valid).sum().item() / len(y_valid)
        valid_loss.append(loss)
        valid_acc.append(acc)

list_epoch = list(range(len(train_loss)))

plt.figure(figsize=(15, 5))
plt.subplot(1,2,1)
plt.plot(list_epoch, train_acc, label="train")
plt.plot(list_epoch, valid_acc, label="validation")
plt.legend()
plt.xlabel("epoch", fontsize = 18)
plt.ylabel("accuracy", fontsize = 18)
plt.tick_params(labelsize=14)

plt.subplot(1,2,2)
plt.plot(list_epoch, train_loss, label="train")
plt.plot(list_epoch, valid_loss, label="validation")
plt.legend()
plt.xlabel("EPOCH", fontsize = 18)
plt.ylabel("loss", fontsize = 18)
plt.tick_params(labelsize=14)

plt.tight_layout()
plt.savefig(f"./data/prob{prob_num}.png")
