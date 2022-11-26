import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from LogisticRegression import LogisticRegression

def main():
    dataset = pd.read_csv("diabetes.csv")
    X = dataset.iloc[:,:-1].values
    y = dataset.iloc[:,-1:].values
    print(X.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
    lr = LogisticRegression(0.01, 1000)
    W, b = lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    print(acc(y_pred, y_test))
    print(y_pred)
    print(y_test.reshape(1,-1))

def acc(y_pred, y_test):
    acc = 0
    for i in range(np.size(y_pred)):
        if (y_pred[i] == y_test[i]):
            acc += 1
    
    return acc/np.size(y_pred)

if __name__ == "__main__":
    main()
