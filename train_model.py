import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

DATA_DIR = "data_custom"  # baseline dataset

X = []
y = []

for label in os.listdir(DATA_DIR):
    label_path = os.path.join(DATA_DIR, label)

    if not os.path.isdir(label_path):
        continue

    for file in os.listdir(label_path):
        if file.endswith(".npy"):
            data = np.load(os.path.join(label_path, file))
            X.append(data)
            y.append(label)

X = np.array(X)
y = np.array(y)

print("Data shape:", X.shape)

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "model_custom.pkl")

print("Model trained and saved!")