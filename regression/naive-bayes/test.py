import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from NaiveBayes import NaiveBayes
from sklearn.model_selection import train_test_split

def main():
    dataset = pd.read_csv("iris.csv")
    X = dataset.loc[:, dataset.columns != "variety"]
    y = dataset.loc[:, dataset.columns == "variety"]
    print(X.head())

    y[y == "Setosa"] = 1
    y[y == "Versicolor"] = 2
    y[y == "Virginica"] = 3

    #print(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = False)
    print(X_train.shape, y_train.shape)
    NB = NaiveBayes()
    class_priors, predictor_priors, likelihoods = NB.fit(X, y)
    print(NB.predict(X_test))
    
    

if __name__ == "__main__":
    main()