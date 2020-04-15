import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

X_train = pd.read_table('./data/train.feature.txt')
y_train = pd.read_table('./data/train.txt')

X_valid = pd.read_table('./data/valid.feature.txt')
y_valid = pd.read_table('./data/valid.txt')

X_test = pd.read_table('./data/test.feature.txt')
y_test = pd.read_table('./data/test.txt')

c_params = [0.1, 1.0, 10.0, 100.0, 1000.0]
train_acc = []
valid_acc = []
test_acc = []

for c in c_params:
    print(f"training with regularization parameter {c}")
    model = LogisticRegression(C=c, solver='lbfgs', random_state=1234)
    model.fit(X_train.values, y_train["CATEGORY"])
    train_acc.append(accuracy_score(y_train["CATEGORY"], model.predict(X_train.values)))
    valid_acc.append(accuracy_score(y_valid["CATEGORY"], model.predict(X_valid.values)))
    test_acc.append(accuracy_score(y_test["CATEGORY"], model.predict(X_test.values)))

plt.figure()
plt.plot(c_params, train_acc, label='train acc')
plt.plot(c_params, valid_acc, label='valid acc')
plt.plot(c_params, test_acc, label='test acc')
plt.legend()
plt.xlabel("regularization parameter")
plt.ylabel("accuracy")
plt.show()