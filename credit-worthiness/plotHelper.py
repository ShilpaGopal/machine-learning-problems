import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')
sns.set_style("darkgrid")


def cust_bar_plot_per(df, x, y, hue=None, title='', width=20, height=8):
    plt.figure(figsize=[width, height])
    ax = sns.barplot(x=x,
                     y=y,
                     data=df,
                     hue=hue,
                     capsize=.5,
                     estimator=lambda x: len(x) / len(df) * 100)
    ax.set(ylabel="Percent")
    ax.set_title(title)
    plt.show()


def cust_bar_plot(df, x, y, hue=None, title='', width=20, height=8):
    plt.figure(figsize=[width, height])
    ax = sns.barplot(x=x,
                     y=y,
                     data=df,
                     hue=hue,
                     capsize=.5)
    ax.set(ylabel="Count")
    ax.set_title(title)
    plt.show()


def cust_cnt_plot(df, x, hue=None, title='', width=20, height=8):
    plt.figure(figsize=[width, height])
    ax = sns.countplot(x=x,
                       data=df,
                       hue=hue)
    ax.set_title(title)
    plt.show()


def customize_scatter_plot(df, y, x, title='', width=20, height=8):
    plt.figure(figsize=[width, height])
    ax = sns.scatterplot(y=df[y], x=df[x])
    ax.set_title(title)
    plt.show()


def customize_strip_plot(df, y, x, title='', width=20, height=8):
    plt.figure(figsize=[width, height])
    ax = sns.stripplot(df[x], df[y])
    ax.set_title(title)
    plt.show()


def customize_box_plot(df, y, x, hue=None, width=20, height=5, title='', orient='v', notch=False, showfliers=True):
    plt.figure(figsize=[width, height])
    if hue:
        ax = sns.boxplot(y=df[y],
                         x=df[x],
                         hue=df[hue],
                         orient=orient,
                         showfliers=showfliers,
                         notch=notch)
    else:
        ax = sns.boxplot(y=df[y],
                         x=df[x],
                         orient=orient,
                         showfliers=showfliers,
                         notch=notch)

    ax.set_title(title)
    plt.show()


def customize_violin_plot(df, y, x, hue=None, orient='v', width=20, height=5, title=''):
    plt.figure(figsize=[width, height])
    ax = sns.violinplot(x=x,
                        y=y,
                        data=df,
                        orient=orient,
                        hue=hue)
    ax.set_title(title)
    plt.show()


def customize_cross_tab(row_list, col_list):
    return pd.crosstab(row_list,
                       col_list,
                       margins=True).style.background_gradient(cmap='summer_r')


def multi_class_dist(train_df, test_df, cv_df, label):
    train_set = []
    cv_set = []
    test_set = []

    train_class_distribution = train_df[label].value_counts()
    test_class_distribution = test_df[label].value_counts()
    cv_class_distribution = cv_df[label].value_counts()

    sorted_train = np.sort(train_class_distribution.values)
    sorted_test = np.sort(test_class_distribution.values)
    sorted_cv = np.sort(cv_class_distribution.values)

    for i in range(len(sorted_train)):
        train_set.append(np.round((train_class_distribution.values[i] / train_df.shape[0] * 100), 3))
    for i in range(len(sorted_test)):
        test_set.append(np.round((test_class_distribution.values[i] / test_df.shape[0] * 100), 3))
    for i in range(len(sorted_cv)):
        cv_set.append(np.round((cv_class_distribution.values[i] / cv_df.shape[0] * 100), 3))

    distribution_per_set = pd.DataFrame(
        {
            'Train Set(%)': train_set,
            'CV Set(%)': cv_set,
            'Test Set(%)': test_set
        })

    # Plotting Distribution per class
    distribution_per_set.index = distribution_per_set.index + 1
    distribution_per_set.plot.bar(figsize=(20, 8))
    plt.xticks(rotation=0)
    plt.title('Distribution of data per set and class')
    plt.xlabel('Class')
    plt.ylabel('% Of total data')
    plt.show()
