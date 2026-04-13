import numpy as np


def analyze_pattern(input_data):
    """
    Intelligent routing based on traffic behavior
    """

    # Try converting numeric values
    numeric_values = []

    for x in input_data:
        try:
            numeric_values.append(float(x))
        except:
            continue

    if len(numeric_values) == 0:
        return "rf"

    avg = np.mean(numeric_values)
    var = np.var(numeric_values)

    # 🔥 REAL LOGIC

    # Low variance → normal/simple → RF
    if var < 100:
        return "rf"

    # Medium variance → complex → DNN
    elif var < 10000:
        return "dnn"

    # High variance → sequential/complex → LSTM
    else:
        return "lstm"