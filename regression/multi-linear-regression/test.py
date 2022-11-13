import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from MultiLinearRegression import MultiLinearRegression

def main():
    dataset = pd.read_csv("estates.csv")
    y = np.array(dataset["Y house price of unit area"]).reshape(-1, 1)
    X = np.array(dataset.drop("Y house price of unit area", axis = 1))

    s = StandardScaler()
    X = s.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,test_size=0.2,random_state=42)

    mlr = MultiLinearRegression(0.01, 1000) #learning rate of 0.01, 1000 epochs
    W, b, costs = mlr.fit(np.array(X_train), y_train)
    y_pred = mlr.predict(X_test)

    r2acc(y_pred, y_test) #runs w/ an r2 accuracy of .6753893104872477 on this dataset

    print(W)
    print(b)
    plt.plot(costs)
    plt.title("Cost function over epochs")
    plt.xlabel("Epoch #")
    plt.ylabel("Cost Function (MSE)")
    plt.show()

def r2acc(y_pred, y_test):
    rss = np.sum((y_pred - y_test) ** 2)
    tss = np.sum((y_test-y_test.mean()) ** 2)
    
    r2 = 1 - (rss / tss)
    print(f"r2 accuracy: {r2}") 

if __name__ == "__main__":
    main()
