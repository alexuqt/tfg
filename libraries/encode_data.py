from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
from . import utils

# previously transform into np array
# one_hot adds columns, so normalized should be called before to avoid mistaking the column numbers

def one_hot(x_train, col_index):
    col = x_train[:, col_index].reshape(-1, 1)
    encoder = OneHotEncoder()
    col = encoder.fit_transform(col)
    x_train = np.concatenate((x_train[:, :col_index], col.toarray(), x_train[:, col_index + 1:]), axis=1)
    print(f"Col {col_index}: One-hot → {col.shape[1]}")
    print(encoder.categories_[0])
    return x_train


def normalize(x_train, col_index):
    col = x_train[:, col_index].astype(float)
    min_value = min(col)
    max_value = max(col)
    col_range = max_value - min_value
    x_train[:, col_index] = [(x - min_value) / col_range for x in col]
    print(f"Col {col_index}: Normalize → 1")
    return x_train


def standardize(x_train, col_index):
    col = x_train[:, col_index].reshape(-1, 1)
    scaler = StandardScaler()
    col = scaler.fit_transform(col)
    x_train[:, col_index] = col.flatten()
    print(f"Col {col_index}: Standardize → 1")
    return x_train


def encode_all(x_train, column_settings, start_time):
    utils.print_time(start_time, "let's encode")

    max_col_index = max(max(col_list, default=-1) for col_list in column_settings)
    for col_index in reversed(range(max_col_index + 1)):
        if col_index in column_settings[0]:
            x_train = normalize(x_train, col_index)
        elif col_index in column_settings[1]:
            x_train = one_hot(x_train, col_index)
        elif col_index in column_settings[2]:
            x_train = standardize(x_train, col_index)
        else:
            print(f"Col {col_index}: -")

    utils.print_time(start_time, "everything encoded")
    return x_train