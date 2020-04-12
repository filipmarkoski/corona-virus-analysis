from collections import namedtuple
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from pandas.api.types import is_numeric_dtype

# Local Imports
from definitions import RNG_SEED


def examine_distribution(sequence, title=''):
    # Calculate statistics
    mean = np.mean(sequence)
    median = np.median(sequence)
    mode = int(scipy.stats.mode(sequence).mode)
    statistics = {'Mean': mean, 'Median': median, 'Mode': mode}

    # Plot distribution
    fig, ax = plt.subplots()
    sns.distplot(sequence)
    ax.axvline(mean, color='r', linestyle='--')
    ax.axvline(median, color='g', linestyle='-')
    ax.axvline(mode, color='b', linestyle='-')

    plt.title(title.capitalize())
    plt.legend(statistics)
    plt.show()

    for key, value in statistics.items():
        print(f'\t{key}={value:.2f}')


def visualize_dict(D, sort=True):
    if sort:
        D = {k: v for k, v in sorted(D.items(),
                                     key=lambda item: item[1])}

    df = pd.DataFrame(data=D, index=list(D.keys()))
    ax = sns.barplot(data=df)

    # adding the text labels
    rects = ax.patches
    labels = list(D.values())
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2,
                height,
                label,
                ha='center',
                va='bottom')

    plt.show()


@contextmanager
def merge_data_frames(x: pd.DataFrame, y: pd.DataFrame):
    data = pd.concat((x, y), ignore_index=True, sort=False)
    yield data
    x = data[:len(x)]
    y = data[len(x):]
    return x, y


def to_one_hot_matrix(predictions_matrix):
    df = pd.DataFrame(data=predictions_matrix)
    for index, row in df.iterrows():
        max_column_index = row.idxmax()
        one_hot_prediction = np.zeros(len(row))
        one_hot_prediction[max_column_index] = 1
        df.iloc[index] = pd.Series(data=one_hot_prediction)

    return df


def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: lime' if v else '' for v in is_max]


def highlight_min(s):
    is_min = s == s.min()
    return ['background-color: lime' if v else '' for v in is_min]


# df.index += 1
# df.astype('int32').style.apply(highlight_non_zeros)
def highlight_non_zeros(s):
    # s stands for pd.Series
    is_zero = (s != 0)
    return ['background-color: yellow' if v else '' for v in is_zero]


def print_sizes(filename, nbytes):
    print(f'`{filename}` in KiB: {(nbytes / 1024) :.2f}')
    print(f'`{filename}` in MiB: {((nbytes / 1024) / 1024):.2f}')
    print(f'`{filename}` in GiB: {((nbytes / 1024) / 1024) / 1024:.2f}')


# From fast.ai
# https://github.com/fastai/fastai/blob/master/old/fastai/structured.py

def fix_missing(df, col, name, na_dict):
    if is_numeric_dtype(col):
        if pd.isnull(col).sum() or (name in na_dict):
            df[name + '_na'] = pd.isnull(col)
            filler = na_dict[name] if name in na_dict else col.median()
            df[name] = col.fillna(filler)
            na_dict[name] = filler
    return na_dict


def numericalize(df, col, name, max_n_cat):
    if not is_numeric_dtype(col) and (max_n_cat is None or len(col.cat.categories) > max_n_cat):
        df[name] = pd.Categorical(col).codes + 1


# From stackoverflow.com

def ndarray_to_dataframe(data: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(data=data[1:, 1:],  # values
                        index=data[1:, 0],  # 1st column as index
                        columns=data[0, 1:])  # 1st row as the column names


# Originals

def df_info(df: pd.DataFrame, verbose: bool = False):
    print(type(df))
    print(df.shape)
    print(df.info(verbose=verbose))
    print(df.head())
    print(df.tail())


def append_isna_columns(df: pd.DataFrame, columns: list):
    for column in columns:
        df[column + '_isna'] = pd.isna(column)
    return df


def split_data_frame(df: pd.DataFrame, percent=0.80, shuffle=True) -> tuple:
    n_rows = df.shape[0]
    split_idx = np.round(n_rows * percent)

    if shuffle:
        top = df[:split_idx].sample(frac=1, random_state=RNG_SEED).reset_index(drop=True)
        bottom = df[split_idx:].sample(frac=1, random_state=RNG_SEED).reset_index(drop=True)
    else:
        top = df[:split_idx].reset_index(drop=True)
        bottom = df[split_idx:].reset_index(drop=True)

    return top, bottom


def split_class(df: pd.DataFrame, class_feature: str) -> tuple:
    y = df[class_feature].copy()
    df.drop(class_feature, axis=1, inplace=True)
    return df, y


def missing_data_ratio(df: pd.DataFrame, display=False):
    # missing data
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    if display:
        print(missing_data.head(20))
    return missing_data


def remove_features_by_missing_data_ratio(df: pd.DataFrame, fraction: float = 0.15, missing_data=None):
    if missing_data is None:
        missing_data = missing_data_ratio(df)
    # dealing with missing data
    df = df.drop((missing_data[missing_data['Percent'] > fraction]).index, 1)
    # df = df.drop(df.loc[df['Electrical'].isnull()].index)
    df.isnull().sum().max()  # just checking that there's no missing data missing...
    return df


def create_namedtuple(X: object, y: object) -> object:
    t = namedtuple('Dataset', ['X', 'y'])
    t.X = X
    t.y = y
    return t


# From hackersandslackers.com

def float_to_int(ser: pd.Series) -> pd.Series:
    try:
        int_ser = ser.astype(int)
        if (ser == int_ser).all():
            return int_ser
        else:
            return ser
    except ValueError:
        return ser


def multi_assign(df, transform_fn, condition):
    df_to_use = df.copy()

    return (df_to_use.assign(
        **{col: transform_fn(df_to_use[col])
           for col in condition(df_to_use)})
    )


def all_float_to_int(df):
    df_to_use = df.copy()
    transform_fn = float_to_int
    condition = lambda x: list(x
                               .select_dtypes(include=["float"])
                               .columns)

    return multi_assign(df_to_use, transform_fn, condition)


def downcast_all(df, target_type, inital_type=None):
    # Gotta specify floats, unsigned, or integer
    # If integer, gotta be 'integer', not 'int'
    # Unsigned should look for Ints
    if inital_type is None:
        inital_type = target_type

    df_to_use = df.copy()

    transform_fn = lambda x: pd.to_numeric(x,
                                           downcast=target_type)

    condition = lambda x: list(x
                               .select_dtypes(include=[inital_type])
                               .columns)

    return multi_assign(df_to_use, transform_fn, condition)


def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    return (df
            .pipe(all_float_to_int)
            .pipe(downcast_all, "float")
            .pipe(downcast_all, "integer")
            .pipe(downcast_all,
                  target_type="unsigned",
                  inital_type="integer")
            )


def main():
    serialized_train_feather = ''
    train: pd.DataFrame = pd.read_feather(serialized_train_feather)

    # Before
    print(train.info(memory_usage=True))

    # After
    train = downcast_dtypes(train)
    print(train.info(memory_usage='deep'))

    numerical_features = train.dtypes[train.dtypes == "float32"].index
    print(len(numerical_features))
    print(numerical_features)


if __name__ == '__main__':
    main()
