import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv("train_landmarks.csv")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ==============================
# MODEL (same as training)
# ==============================
model = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    max_iter=200,
    early_stopping=True
)

# ==============================
# CROSS VALIDATION
# ==============================
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

print("\n==============================")
print("Cross Validation Accuracy:")
print(scores)
print("------------------------------")
print(f"Mean Accuracy: {scores.mean():.4f}")
print(f"Std Deviation: {scores.std():.4f}")
print("==============================")