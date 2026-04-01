import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import joblib

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv("train_landmarks.csv")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# ==============================
# ENCODE LABELS
# ==============================
le = LabelEncoder()
y = le.fit_transform(y)

print("Classes:", le.classes_)

# ==============================
# SCALE FEATURES
# ==============================
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ==============================
# TRAIN / VALIDATION SPLIT
# ==============================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

# ==============================
# CLASS WEIGHTS (IMPORTANT)
# ==============================
classes = np.unique(y_train)
weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))

print("\nClass Weights:")
for k, v in class_weights.items():
    print(f"Class {k}: {v:.3f}")

# Convert to sample weights
sample_weights = np.array([class_weights[label] for label in y_train])

# ==============================
# MODEL
# ==============================
model = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation='relu',
    solver='adam',
    max_iter=300,
    early_stopping=True,
    n_iter_no_change=10,
    verbose=True
)

# ==============================
# TRAIN
# ==============================
model.fit(X_train, y_train, sample_weight=sample_weights)

# ==============================
# EVALUATE
# ==============================
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)

train_acc = accuracy_score(y_train, y_train_pred)
val_acc = accuracy_score(y_val, y_val_pred)

print("\n==============================")
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
print("==============================")

# ==============================
# SAVE EVERYTHING
# ==============================
joblib.dump(model, "model.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ Model, encoder, and scaler saved!")