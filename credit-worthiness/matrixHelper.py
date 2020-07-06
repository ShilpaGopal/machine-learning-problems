from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


def precision(y_test, predicted_y, labels):
    return precision_score(y_test, predicted_y, labels=labels, average='micro')


def recall(y_test, predicted_y):
    return recall_score(y_test, predicted_y, average='micro')


def f1score(y_test, predicted_y):
    return f1_score(y_test, predicted_y, average='micro')


def plot_matrices(y_test, predicted_y, labels):
    confusion = confusion_matrix(y_test, predicted_y)
    precision = (confusion / confusion.sum(axis=0))
    recall = (confusion.T / (confusion.sum(axis=1))).T

    f, (ax1, ax2, ax3, axcb) = plt.subplots(1, 4,
                                            gridspec_kw={'width_ratios': [1, 1, 1, 0.05]}, figsize=(22, 6))

    g1 = sns.heatmap(confusion, cbar=False, ax=ax1, annot=True, cmap="YlGnBu", linewidths=.5, fmt="d",
                     xticklabels=labels,
                     yticklabels=labels, )
    g1.set_ylabel('Original Class')
    g1.set_xlabel('Predicted Class')
    g1.set_title('Confusion')

    g2 = sns.heatmap(precision, cmap="YlGnBu", cbar=False, ax=ax2, annot=True, fmt=".3f",
                     xticklabels=labels,
                     yticklabels=labels)

    g2.set_ylabel('Original Class')
    g2.set_xlabel('Predicted Class')
    g2.set_title('Precision')

    g3 = sns.heatmap(recall, cmap="YlGnBu", ax=ax3, cbar_ax=axcb, annot=True, fmt=".3f",
                     xticklabels=labels,
                     yticklabels=labels)

    g3.set_ylabel('Original Class')
    g3.set_xlabel('Predicted Class')
    g3.set_title('Recall')

    for ax in [g1, g2, g3]:
        tl = ax.get_xticklabels()
        ax.set_xticklabels(tl, rotation=0)
        tly = ax.get_yticklabels()
        ax.set_yticklabels(tly, rotation=0)

    plt.show()


