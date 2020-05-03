import joblib
import numpy as np
import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, vector_size, output_size=4):
        super(NeuralNet, self).__init__()
        self.linear = nn.Linear(vector_size, output_size, bias=False)
        self.out = nn.Softmax(dim=1)
    
    def forward(self, x):
        h = self.linear(x)
        out = self.out(h)
        return out

X_train = np.array(list(joblib.load('./data/X_train.joblib')))
X_train = torch.from_numpy(X_train.astype(np.float32))
X = X_train[0:4]

model = NeuralNet(X_train.shape[1], 4)
y_pred = model(X)
print(y_pred)