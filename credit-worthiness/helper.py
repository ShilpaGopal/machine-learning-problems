import numpy as np
import pandas as pd


def missing_data_points(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data


def outlier_treatment(data_column):
    sorted(data_column)
    Q1, Q3 = np.nanpercentile(data_column, [25, 75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    return lower_range, upper_range


def count_percentage_mean(df, df_column, label):
    count = df.groupby(df_column).size()
    percent = df.groupby(df_column).size() / len(df)
    mean_loan_amount = df.groupby(df_column)[label].mean()
    table = pd.concat([count, percent, mean_loan_amount], axis=1, keys=['count', 'percentage', 'mean'])
    return table


def count_percentage_mode(df, df_column, label):
    count = df.groupby(df_column).size()
    percent = df.groupby(df_column).size() / len(df)
    mode_loan_amount = df.groupby(df_column)[label].agg(lambda x: pd.Series.mode(x)[0])
    table = pd.concat([count, percent, mode_loan_amount], axis=1, keys=['count', 'percentage', 'mode'])
    return table


def count_percentage(df, df_column):
    count = df.groupby(df_column).size()
    percent = df.groupby(df_column).size() / len(df)
    table = pd.concat([count, percent], axis=1, keys=['count', 'percentage'])
    return table


def convert_numeric(df, column):
    labels = df[column].astype('category').cat.categories.tolist()
    replace_labels = {column: {k: v for k, v in zip(labels, list(range(1, len(labels) + 1)))}}
    df.replace(replace_labels, inplace=True)
    return df