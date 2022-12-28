import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

class NaiveBayes:
    def fit(self, X, y):
        self.X = X
        self.y = y
        n, m = X.shape
        num_classes = len(np.unique(y))

        self.class_priors = {}
        self.predictor_priors = {}
        self.likelihoods = {}

        for feature in list(self.X.columns):
            for feat_val in np.unique(self.X[feature]):
                self.predictor_priors[str(feature) + "_" + str(feat_val)] = 0
                for outcome in np.unique(self.y):
                    self.likelihoods[str(feature) + "_" + str(feat_val) + '_' + str(outcome)] = 0
                    self.class_priors[outcome] = 0

        #calculating prior probabilities of classes => P(y)
        for elem in np.unique(y):
            self.class_priors[elem] = len(np.where(self.y == elem)) / m

        #calculating prior probabilities of predictors => P(x)
        for x in list(self.X.columns): #for each feature in X
            feature_dict = {elem: 0. for elem in set(X[x])} #inits a dictionary to count P(x) for each unique element in the column
            for value in list(X[x]):
                feature_dict[value] += 1/n #adds 1/n per unique value -> n is # of samples
                self.predictor_priors[x + "_" + str(value)] = feature_dict[value] #adds this dictionary to a greater dictionary for the predictions

        #calculating likelihoods => P(x|y)
        for x in list(self.X.columns):
            for outcome in np.unique(y):
                list_matches = np.where(y == outcome)[0]
                feat_likelihood = X[x][list_matches].value_counts().to_dict()
                for feat_val, count in feat_likelihood.items():
                    self.likelihoods[x + "_" + str(feat_val) + "_" + str(outcome)] = count/len(list_matches)

        return self.class_priors, self.predictor_priors, self.likelihoods

    #calculates posterior probability => P(y|x)
    def predict(self, X_test):
        results = []
        X_test = np.array(X_test)

        for x in X_test:
            probs = {}
            for elem in np.unique(np.array(self.y)):
                prior = self.class_priors[elem]
                likelihood = 1
                evid = 1

                for ft, ft_val in zip(list(self.X.columns), x):
                    likelihood *= self.likelihoods[ft + "_" + str(ft_val) + "_" + str(elem)]
                    evid *= self.predictor_priors[ft + "_" + str(ft_val)]

                posterior = likelihood*prior/evid
                probs[elem] = posterior

            result = max(list(probs.values()))
            results.append(result)
        
        return np.array(results)