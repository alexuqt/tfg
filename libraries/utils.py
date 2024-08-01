import time


def prints(x_train, y_train, rows_to_print):
    print("---")
    for i in range(0, rows_to_print):
        print(f"x_train[{i}]:", x_train[i])
        print(f"y_train[{i}]:", y_train[i])
    print("")


def print_time(start_time, msg):
    new_time = time.time()
    formatted_time = time.strftime('%H:%M', time.localtime(new_time))
    print(
        f"\n--- ({formatted_time}) {msg} {round(new_time - start_time)} seconds ({round((time.time() - start_time) / 60)} minutes) ---")
    return new_time


def print_summary(x_train, x_val, y_train, rows_to_print, start_time):
    print("\n---------------\n--- SUMMARY ---\n---------------\n")
    print(f'x_train rows: {x_train.shape[0]}')
    print(f'x_val rows: {x_val.shape[0]}')

    prints(x_train, y_train, rows_to_print)
    print_time(start_time, "Preprocessing")


def top_x(filename, element):
    with open(f'../tops/{filename}', 'r') as file:
        elements = file.read().splitlines()
    return element in elements


def print_header(x):
    headers = {
        39: 'model final'
    }
    print(f'data code {x}: {headers.get(x, "no info / wrong data code")}')


def datafile(code):
    if code < 10:
        return f'0{code}.csv'
    else:
        return f'{code}.csv'
