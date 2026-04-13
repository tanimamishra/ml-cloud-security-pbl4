import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# ===============================
# LOAD DATA
# ===============================
train = pd.read_csv("../dataset/nsl_kdd_train.csv")
test = pd.read_csv("../dataset/nsl_kdd_test.csv")

print("Data loaded ✅")


# ===============================
# LABEL CONVERSION
# ===============================
train['label'] = train['label'].apply(lambda x: 0 if x == "normal" else 1)
test['label'] = test['label'].apply(lambda x: 0 if x == "normal" else 1)


# ===============================
# SPLIT FEATURES
# ===============================
X = train.drop(columns=['label'])
y = train['label']

X_test = test.drop(columns=['label'])
y_test = test['label']


# ===============================
# ENCODING
# ===============================
X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)

# Align test with train
X_test = X_test.reindex(columns=X.columns, fill_value=0)

print("Encoding done ✅")


# ===============================
# SAVE FEATURE COLUMNS (CRITICAL)
# ===============================
feature_columns = X.columns.tolist()


# ===============================
# SCALING
# ===============================
scaler = StandardScaler()

X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

print("Scaling done ✅")


# ===============================
# SPLIT
# ===============================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ===============================
# MODEL
# ===============================
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Model ready ✅")


# ===============================
# TRAIN
# ===============================
model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=128,
    validation_data=(X_val, y_val),
    verbose=1
)


# ===============================
# TEST
# ===============================
y_pred = (model.predict(X_test) > 0.3).astype(int)

acc = accuracy_score(y_test, y_pred)

print(f"\n🔥 FINAL DNN Accuracy: {acc}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ===============================
# SAVE EVERYTHING (FINAL FIX)
# ===============================
os.makedirs("models", exist_ok=True)

model.save("models/dnn_model.h5")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(feature_columns, "models/feature_columns.pkl")

print("\nAll files saved correctly ✅")