from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold
from . import utils
import numpy as np
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

def begin(x_train, y_train, x_val, y_val, hidden_layers, epochs, num_outputs, start_time, x_val_readable, data_code):
    model = create_model(x_train.shape[1:], num_outputs, hidden_layers)
    model.summary()
    model.fit(x_train, y_train, epochs=epochs, batch_size=32, validation_data=(x_val, y_val))
    utils.print_time(start_time, "Training done")

    predict(x_val.shape[0], model, x_val, y_val, x_val_readable, data_code, num_outputs)

    save_weights(model, data_code)
    utils.print_time(start_time, "Finished after")


def create_model(input_shape, num_outputs, hidden_layers):
    model = Sequential()

    if hidden_layers == 0:
        model.add(Dense(num_outputs, activation='linear', input_shape=input_shape))

    else:
        model.add(Dense(round(input_shape[0]/2), activation='relu', input_shape=input_shape))
        for _ in range(hidden_layers - 1):
            model.add(Dense(round(input_shape[0]/2), activation='relu'))
        model.add(Dense(num_outputs, activation='linear'))

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


def predict(num, model, x_val, y_val, x_val_readable, data_code, num_outputs):

    predictions = model.predict(x_val[:num])
    res = y_val[:num]
    with open(f'predictions-{data_code}.txt', 'w') as f:
        if num_outputs == 1:
            for i in range(num):
                x_val_str = ','.join(map(str, x_val_readable[i]))
                if i < 10:
                    print(str(res[i]) + " -> " + str(predictions[i][0]))
                f.write(f"{x_val_str},{res[i]},{predictions[i][0]}\n")
        else:
            for i in range(num):
                x_val_str = ','.join(map(str, x_val_readable[i]))
                actual = ','.join(map(str, res[i]))
                predicted = ','.join([f"{value:.2f}" for value in predictions[i]])
                if i < 10:
                    print(str(actual) + " -> " + str(predicted))
                f.write(f"{x_val_str},{actual},{predicted}\n")


def save_weights(model, data_code):

    with open(f'weights-{data_code}.txt', 'w') as f:
        for layer in range(len(model.layers)):
            f.write(f"LAYER {layer}\n")

            weights, biases = model.layers[layer].get_weights()

            f.write("Weights\n")
            for weight in weights:
                f.write(f"{weight[0]}\n")

            f.write(f"Biases\n{biases}")
            f.write("Biases\n")
            for bias in biases:
                f.write(f"{bias}\n")


def k_fold_cross_validation(x_train, y_train, hidden_layers, num_outputs, epochs, start_time, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = []
    times = []

    new_time = utils.print_time(start_time, f'Starting cross-validation. Time updated ---')
    for fold, (train_index, test_index) in enumerate(kf.split(x_train), 1):
        x_train_fold, x_val_fold = x_train[train_index], x_train[test_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[test_index]

        num_rows_fold = len(test_index)

        model = create_model(x_train.shape[1:], num_outputs, hidden_layers)
        model.fit(x_train_fold, y_train_fold, epochs=epochs, batch_size=32, verbose=0)
        score = model.evaluate(x_val_fold, y_val_fold, verbose=0)
        scores.append(score)
        old_time = new_time
        new_time = utils.print_time(new_time, f'Fold {fold}/{k} ({num_rows_fold} rows) - MSE: {round(score,3)} ---')
        times.append (new_time-old_time)

    mean_mse = np.mean(scores)
    std_mse = np.std(scores)
    avg_time = np.mean(times)
    minutes = round(avg_time//60)
    seconds = round(avg_time%60)
    print(f'K-Fold Cross-Validation - Mean MSE: {round(mean_mse, 3)}, Std: {round(std_mse, 3)}, Avg Time: {minutes} min {seconds} s')
    return mean_mse, std_mse
