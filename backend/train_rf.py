import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


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
# BALANCE DATA (IMPORTANT 🔥)
# ===============================
df = train.copy()

df_normal = df[df['label'] == 0]
df_attack = df[df['label'] == 1]

df_attack_upsampled = resample(
    df_attack,
    replace=True,
    n_samples=len(df_normal),
    random_state=42
)

df_balanced = pd.concat([df_normal, df_attack_upsampled])


# ===============================
# SPLIT
# ===============================
X = df_balanced.drop(columns=['label'])
y = df_balanced['label']

X_test = test.drop(columns=['label'])
y_test = test['label']


# ===============================
# ENCODING
# ===============================
X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)

X_test = X_test.reindex(columns=X.columns, fill_value=0)


# ===============================
# TRAIN RF (TUNED)
# ===============================
model = RandomForestClassifier(
    n_estimators=800,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

print("Training done ✅")


# ===============================
# EVALUATION
# ===============================
y_prob = model.predict_proba(X_test)[:, 1]

# Lower threshold (important)
y_pred = (y_prob > 0.35).astype(int)

print("\n🔥 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))


# ===============================
# SAVE MODEL
# ===============================
os.makedirs("../backend/models", exist_ok=True)

joblib.dump(model, "../backend/models/rf_model.pkl")
joblib.dump(X.columns.tolist(), "../backend/models/feature_columns.pkl")

print("\nModel saved ✅")