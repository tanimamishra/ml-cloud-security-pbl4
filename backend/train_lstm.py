import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input


# ===============================
# LOAD DATA
# ===============================
train = pd.read_csv("../dataset/nsl_kdd_train.csv")
test = pd.read_csv("../dataset/nsl_kdd_test.csv")

print("Data loaded ✅")


# ===============================
# FIX LABELS (binary)
# ===============================
def convert_label(x):
    return 0 if x == "normal" else 1


train['label'] = train['label'].apply(convert_label)
test['label'] = test['label'].apply(convert_label)

print("Labels converted ✅")


# ===============================
# SPLIT FEATURES & TARGET
# ===============================
X_train = train.drop(columns=['label'])
y_train = train['label']

X_test = test.drop(columns=['label'])
y_test = test['label']


# ===============================
# ENCODE CATEGORICAL FEATURES
# ===============================
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Align
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

print("Feature alignment done ✅")


# ===============================
# SCALING
# ===============================
scaler = joblib.load("../backend/models/scaler.pkl")

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print("Scaling applied ✅")


# ===============================
# RESHAPE FOR LSTM
# ===============================
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])


# ===============================
# BUILD MODEL (BiLSTM)
# ===============================
model = Sequential()

model.add(Input(shape=(1, X_train.shape[2])))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.3))

model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()


# ===============================
# TRAIN
# ===============================
print("Training BiLSTM...")

model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)


# ===============================
# EVALUATE
# ===============================
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)

print(f"✅ LSTM Accuracy: {accuracy}")


# ===============================
# SAVE MODEL
# ===============================
model.save("../backend/models/lstm_model.h5")

print("BiLSTM model saved ✅")