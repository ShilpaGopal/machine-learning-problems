import numpy as np
from sklearn.metrics import confusion_matrix


def custom_weighted_loss(y_test, predicted_y):
    confusion = confusion_matrix(y_test, predicted_y, labels=[2000, 5000, 7000, 8000, 10000, 12000, 15000])
    n_classes = confusion.shape[0]
    sum0 = np.sum(confusion, axis=0)
    sum1 = np.sum(confusion, axis=1)
    expected = np.outer(sum0, sum1) / np.sum(sum0)
    w_mat = np.zeros([n_classes, n_classes], dtype=np.float)
    for i in range(len(w_mat)):
        for j in range(len(w_mat)):
            if i < j:
                w_mat[i][j] = 3

    for i in range(len(w_mat)):
        for j in range(len(w_mat)):
            if i > j:
                w_mat[i][j] = float(((i - j) ** 2) / 16)

    k = np.sum(w_mat * confusion) / np.sum(w_mat * expected)
    return k


def predict_from_probability(probability, labels):
    class_index = probability.argmax(axis=-1)
    class_map = {i: labels[i] for i in range(0, len(labels))}
    predictions = [class_map[k] for k in class_index]
    return np.array(predictions)


def weighted_agreement_score(y_test, predicted_y, labels):
    confusion = confusion_matrix(y_test, predicted_y, labels=labels)
    n_classes = confusion.shape[0]
    sum0 = np.sum(confusion, axis=0)
    sum1 = np.sum(confusion, axis=1)
    expected = np.outer(sum0, sum1) / np.sum(sum0)
    w_mat = np.zeros([n_classes, n_classes], dtype=np.float)
    for i in range(len(w_mat)):
        for j in range(len(w_mat)):
            if i < j:
                w_mat[i][j] = 3

    for i in range(len(w_mat)):
        for j in range(len(w_mat)):
            if i > j:
                w_mat[i][j] = float(((i - j) ** 2) / 16)
    k = np.sum(w_mat * confusion) / np.sum(w_mat * expected)
    return 1-k

