import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from SVM import SVM

def main():
    dataset = pd.read_csv("estates.csv")
    X = np.array(dataset["X2 house age"]).reshape(-1,1)
    y = np.array(dataset["Y house price of unit area"]).reshape(-1,1)
    print(X.shape)

    # Classes 1 and -1
    y = np.where(y == 0, -1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    svm = SVM()
    W, b, losses = svm.fit(X_train, y_train, 100, 0.001, 1000)
    prediction = svm.predict(X_test)

    print(f"Y_pred: {prediction}")
    print(f"Accuracy: {accuracy_score(prediction, y_test)}")
    print(f"Weight: {W}\nBias:{b}")

    plt.plot(losses)
    plt.title("Loss function over epochs")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss Function (Hinge Loss)")
    plt.show()

if __name__ == "__main__":
    main()