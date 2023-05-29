import os
import math
import wandb
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, confusion_matrix, classification_report


def train_valid_split(data_set, valid_ratio, seed):
    """Split provided training data into training set and validation set"""
    valid_set_size = int(valid_ratio * len(data_set))
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)


def print_data(data):
    balance_check = dict(data["target"].value_counts())
    print(f"Total number of data: {data.shape[0]}")
    print(f"Total number of features: {data.shape[1]}")
    print(f"Number of people not looking for job change: {balance_check[0]}")
    print(f"Number of people looking for a job change: {balance_check[1]}")
    print(data.values)

    numpy_data = data.values
    train_data, valid_data = train_valid_split(numpy_data, valid_ratio=0.2, seed=724)
    print(f"Training data: {train_data.shape[0]}")
    print(f"Validation data: {valid_data.shape[0]}")


# Read and print data
data = pd.read_csv("aug_train_preprocessed_onehot.csv")
print_data(data)


# Obtain X and y
columns_to_exclude = ["target"]
X_data = data.drop(columns=columns_to_exclude)
X, y = X_data, data["target"]


# SVC (Split data by order 80% vs. 20%)
position = int(len(X) * 80 / 100)

X_train, y_train = X[:position], y[:position]
X_test, y_test = X[position:], y[position:]

clf = svm.SVC(kernel="linear", random_state=2023)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_train)
print("Train clf.score = ", clf.score(X_train, y_train))
print("Train AC = ", accuracy_score(y_pred, y_train))
print()

y_pred = clf.predict(X_test)
print("Test clf.score = ", clf.score(X_test, y_test))
print("Test AC = ", accuracy_score(y_pred, y_test))

clf = svm.SVC(kernel="linear", random_state=2023)
clf.fit(X, y)
y_pred = clf.predict(X)
conf_matrix = confusion_matrix(y, y_pred)


# Print the confusion matrix using Matplotlib
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i, s=conf_matrix[i, j], va="center", ha="center", size="xx-large")

plt.xlabel("Predictions", fontsize=18)
plt.ylabel("Actuals", fontsize=18)
plt.title("Confusion Matrix", fontsize=18)
plt.savefig("figures/confusion_matrix.png")
plt.show()


print(classification_report(y, y_pred))


# SVC(Split data randomly from 95~5%)
ratio = 100
ratiovalues = [i for i in range(5, ratio, 5)]
train_scores = []
test_scores = []

for i in ratiovalues:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i / 100, random_state=2023)

    clf = svm.SVC(random_state=2023)  # change model
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    train_acc = accuracy_score(y_pred_train, y_train)
    train_scores.append(train_acc)

    y_pred_test = clf.predict(X_test)
    test_acc = accuracy_score(y_pred_test, y_test)
    test_scores.append(test_acc)

    print(">%d, train: %.3f, test: %.3f" % (i, train_acc, test_acc))

plt.plot(ratiovalues, train_scores, "-o", label="Train")
plt.plot(ratiovalues, test_scores, "-o", label="Test")
plt.legend()
plt.savefig("figures/accuracy_plot.png")
plt.show()


# Sort important feature or feature importances by importances
def plot_feature_importances(model, feature_names, top_n=5):
    n_features = len(feature_names)
    importances = np.abs(model.coef_[0])
    indices = np.argsort(importances)
    top_indices = indices[-top_n:]
    top_importances = importances[top_indices]
    top_feature_names = feature_names[top_indices]
    plt.barh(np.arange(top_n), top_importances, align="center")
    plt.yticks(np.arange(top_n), top_feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, top_n)
    plt.savefig("figures/feature_importance.png")
    plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023)

clf = svm.SVC(kernel="linear", random_state=2023)
clf.fit(X_train, y_train)

feature_names = X.columns
plot_feature_importances(clf, feature_names, top_n=5)
