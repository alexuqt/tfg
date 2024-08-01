import time
import numpy as np
from libraries import encode_data, load_data, utils, nn_model


start_time = time.time()
data_code = 39 # el codi és per escalabilitat; el 39 és el que s'utilitza per obtenir els resultats
epochs = 5
hidden_layers = 3

# Load and encode data
x_train, y_train, _, column_settings, split, until, num_outputs = load_data.x(data_code, 1)
#x_val_readable = np.hstack((x_train, days.reshape(-1, 1)))
x_train = encode_data.encode_all(x_train, column_settings, start_time)
x_train, y_train, x_val, _ = load_data.split(x_train, y_train, split, until, True, True, start_time)

utils.print_summary(x_train, x_val, y_train, 1, start_time)

# Perform K-Fold Cross-Validation
mean_mse, std_mse = nn_model.k_fold_cross_validation(x_train, y_train, hidden_layers, num_outputs, epochs, start_time)

