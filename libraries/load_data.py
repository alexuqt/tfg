import csv
import numpy as np
from datetime import datetime
from . import utils


def x(x, month):
    utils.print_header(x)

    month1 = f'0{month}' if month < 10 else f'{month}'
    month2 = f'0{month+1}' if month+1 < 10 else f'{month+1}'

    x_data = []
    y_data = []
    days = []
    split = -1
    until = -1

    filename = utils.datafile(x)

    with open(f'dades/{filename}', 'r', newline='') as csvfile:
        delimiter = ','
        csv_reader = csv.reader(csvfile, delimiter=delimiter)
        found = False
        found_until = False
        count = -1
        key = x
        if x == 3:
            key = 2
        elif x == 8:
            key = 7
        elif x == 26 or x == 27:
            key = 24
        elif x == 33:
            key = 32
        elif x == 36 or x == 37:
            key = 34
        for row in csv_reader:
            result = globals()[f'x{key}_row'](row)
            if result is not None:
                x_row, y_row, day = result
                x_data.append(x_row)
                y_data.append(y_row)
                days.append(day)
                count = count + 1
            if row[0] == f'2018/{month1}/01' and not found:
                split = count + 1
                found = True
            if row[0] == f'2018/{month2}/01' and not found_until:
                until = count
                found_until = True
                print(f'data will be splitted at position {split} until position {until}')

    x_data = np.array(x_data)
    y_data = np.array(y_data)
    days = np.array(days)
    column_settings = columns(x)
    num_outputs = 1 if len(y_data.shape) == 1 else y_data.shape[1]

    utils.prints(x_data, y_data, 1)

    return x_data, y_data, days, column_settings, split, until, num_outputs


def x39_row(row):
    # 2013/01/01,0.008,112,0.395833333,9,1,MIA,L,P,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,North America,0
    year, month, day = map(int, row[0].split("/"))
    #if year != 2018 and year != 2017 and year != 2016:
    #    return None
    weekday = datetime(year, month, day).weekday()
    airline = (row[1]).encode('utf-8')[:5]
    time = float(row[3])
    operation = np.int8(row[5])
    airport = (row[6]).encode('utf-8')[:4]
    fundido = 1 if row[8] == "F" else 0
    preaviso = int(row[9])
    pmrs = int(row[32])

    # not used
    flight_number = row[2]
    hour = int(row[4])
    country = (row[33]).encode('utf-8')[:24]
    airspace = (row[34]).encode('utf-8')[:15]
    blnd = int(row[10])
    deaf = int(row[11])
    dpna = int(row[12])
    wchc = int(row[13])
    wchr = int(row[14])
    wchs = int(row[15])
    maas = int(row[16])
    meda = int(row[17])
    stcr = int(row[18])
    wchp = int(row[19])
    desconocido = int(row[20])
    preaviso_blnd = int(row[21])
    preaviso_deaf = int(row[22])
    preaviso_dpna = int(row[23])
    preaviso_wchc = int(row[24])
    preaviso_wchr = int(row[25])
    preaviso_wchs = int(row[26])
    preaviso_maas = int(row[27])
    preaviso_meda = int(row[28])
    preaviso_stcr = int(row[29])
    preaviso_wchp = int(row[30])
    preaviso_desconocido = int(row[31])

    return (year, month, weekday, airline, time, operation, airport, fundido, preaviso), (pmrs), day


def columns(x):
    # columns_normalize, columns_one_hot, columns_standardize
    column_settings = {
        39: ([0], [6], [])
    }
    return column_settings.get(x, (None, None, None))


def split(x_train, y_train, split, until, validation_different, add_validation, start_time):
    x_train = x_train.astype(float)
    y_train = y_train.astype(float)
    utils.print_time(start_time, "floated")

    x_val = x_train[split:until] if validation_different else x_train[:split]
    y_val = y_train[split:until] if validation_different else y_train[:split]
    x_train = np.concatenate((x_train[:split], x_train[until:]))
    y_train = np.concatenate((y_train[:split], y_train[until:]))

    if add_validation:
        x_train = np.concatenate((x_train, x_val))
        y_train = np.concatenate((y_train, y_val))

    return x_train, y_train, x_val, y_val
