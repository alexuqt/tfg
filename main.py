import time
import numpy as np
from libraries import encode_data, load_data, nn_model, utils

start_time = time.time()
rows_to_print = 1

data_code = 39
epochs = 15
validation_different = True
add_validation = False
hidden_layers = 3
month = 1

# ------------------------------

x_train, y_train, days, column_settings, split, until, num_outputs = load_data.x(data_code, month)
x_val_readable = np.hstack((x_train, days.reshape(-1, 1)))

x_train = encode_data.encode_all(x_train, column_settings, start_time)
x_train, y_train, x_val, y_val = load_data.split(x_train, y_train, split, until, validation_different, add_validation, start_time)
x_val_readable = x_val_readable[split:until] if validation_different else x_val_readable[:split]

utils.print_summary(x_train, x_val, y_train, rows_to_print, start_time)

nn_model.begin(x_train, y_train, x_val, y_val, hidden_layers, epochs, num_outputs, start_time, x_val_readable, data_code)

