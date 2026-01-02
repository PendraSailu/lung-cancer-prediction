import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Load dataset
data = pd.read_csv("data/lungcancer.csv")

# Encode
le = LabelEncoder()
data['GENDER'] = le.fit_transform(data['GENDER'])
data['LUNG_CANCER'] = le.fit_transform(data['LUNG_CANCER'])

X = data.drop('LUNG_CANCER', axis=1)
y = data['LUNG_CANCER']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Best K
k_values = range(1, 21)
acc_list = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knn.fit(X_train, y_train)
    acc_list.append(accuracy_score(y_test, knn.predict(X_test)))

best_k = k_values[np.argmax(acc_list)]

# Train final model
model = KNeighborsClassifier(n_neighbors=best_k, weights="distance")
model.fit(X_train, y_train)

# Create model directory if not exists
os.makedirs("model", exist_ok=True)

# Save files
joblib.dump(model, "model/knn_lung_cancer.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("✅ Model and scaler saved successfully")
