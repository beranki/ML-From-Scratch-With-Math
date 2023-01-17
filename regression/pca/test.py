import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PCA import PCA

def main():
    dataset = pd.read_csv("iris.csv")
    X = dataset.loc[:, dataset.columns != "variety"]
    y = dataset.loc[:, dataset.columns == "variety"]
    print(X.head())

    y[y == "Setosa"] = 1
    y[y == "Versicolor"] = 2
    y[y == "Virginica"] = 3

    pca = PCA(n_components = 2)
    pca_stats = pca.fit(X)
    print(pca_stats)

    X_proj = pca.transform(X)
    print([X_proj[0]])    

    plt.scatter(X_proj[0], X_proj[1], c = np.array(y))
    plt.xlabel("PC1")
    plt.xticks([])
    plt.ylabel("PC2")
    plt.yticks([])

    plt.title("Captures {}% \of variance".format(pca_stats["Cumulative Explained Variance"][1] * 100))
    plt.show()

if __name__ == "__main__":
    main()