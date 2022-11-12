import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression

def main():
    dataset = pd.read_csv("estates.csv")
    X = np.array(dataset["X2 house age"]).reshape(-1,1)
    y = np.array(dataset["Y house price of unit area"]).reshape(-1,1)
    print(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,test_size=0.2,random_state=100)
    lr = LinearRegression(0.01, 1000)
    b_0, b_1 = lr.fit(X_train, y_train)

    plt.scatter(X, y)
    y_pred = b_0 + b_1*X
    plt.plot(X, y_pred, color = "m")
    plt.show()

if __name__ == "__main__":
    main()
