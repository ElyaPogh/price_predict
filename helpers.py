import math

def transformation(column):
    max_value = column.max()
    sin_values = [math.sin((2 * math.pi * x) / max_value) for x in list(column)]
    cos_values = [math.cos((2 * math.pi * x) / max_value) for x in list(column)]
    return sin_values, cos_values


def back_to_back(pred, df):

    return abs(int(pred * df.std() + df.mean()))

def shape_normalization(model, df):
    shape = model.layers[0].output_shape[1]
    params = model.layers[0].count_params()
    columns_count = int(params / shape) - 1
    for i in range(0, max(0, columns_count - len(list(df)))):
        df[i] = 0
    return df