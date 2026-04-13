import joblib
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model

# ===============================
# LOAD FILES
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

dnn_model = load_model(os.path.join(BASE_DIR, "models/dnn_model.h5"))
scaler = joblib.load(os.path.join(BASE_DIR, "models/scaler.pkl"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "models/feature_columns.pkl"))

print("DNN model loaded ✅")


# ===============================
# PREPROCESS
# ===============================
def preprocess_input(input_data):

    df = pd.DataFrame([input_data])
    df = pd.get_dummies(df)

    # Align EXACT features
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Scale
    df_scaled = scaler.transform(df)

    return df_scaled


# ===============================
# FINAL PREDICTION
# ===============================
def route_and_predict(input_data):

    try:
        # ---------------------------
        # RULE-BASED DETECTION
        # ---------------------------
        try:
            src_bytes = float(input_data[4])
            dst_bytes = float(input_data[5])
        except:
            src_bytes, dst_bytes = 0, 0

        if src_bytes > 4000 and dst_bytes > 8000:
            return {
                "prediction": "Attack",
                "model_used": "Rule-Based (Traffic Spike)"
            }

        if src_bytes == 0 and dst_bytes == 0:
            return {
                "prediction": "Attack",
                "model_used": "Rule-Based (Null Traffic)"
            }

        if str(input_data[1]).lower() == "icmp":
            return {
                "prediction": "Attack",
                "model_used": "Rule-Based (ICMP Activity)"
            }

        # ---------------------------
        # DNN PREDICTION
        # ---------------------------
        input_processed = preprocess_input(input_data)

        prob = dnn_model.predict(input_processed)[0][0]

        prediction = "Attack" if prob > 0.3 else "Normal"

        return {
            "prediction": prediction,
            "model_used": "DNN"
        }

    except Exception as e:
        return {
            "prediction": "Error",
            "model_used": str(e)
        }