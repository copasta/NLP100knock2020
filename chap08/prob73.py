import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from tqdm import tqdm


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
X_train = torch.from_numpy(X_train.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.int64))

X = X_train
y = y_train

model = NeuralNet(X_train.shape[1], 4)
optimizer = SGD(model.parameters(), lr=0.01)
loss_hist = []

for epoch in tqdm(range(100)):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = loss_fn(y_pred, y)

    loss.backward()
    optimizer.step()
    
    loss_hist.append(loss)

plt.figure()
plt.plot(list(range(len(loss_hist))), loss_hist)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig("./data/prob73.png")
