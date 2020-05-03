import joblib
import numpy as np
import torch
import torch.nn as nn

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
X_train = torch.from_numpy(X_train.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.int64))

X = X_train[0:4]
y = y_train[0:4]

model = NeuralNet(X_train.shape[1], 4)
y_pred = model(X)
print(y_pred)
loss = loss_fn(y_pred, y)
print(loss)