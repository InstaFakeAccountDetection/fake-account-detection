import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv("train.csv")

# ==========================
# FEATURE ENGINEERING
# ==========================

df["ff_ratio"] = df["#followers"] / (df["#follows"] + 1)
df["posts_per_follower"] = df["#posts"] / (df["#followers"] + 1)
df["has_description"] = df["description length"].apply(lambda x: 1 if x > 0 else 0)

# ==========================
# SPLIT FEATURES & TARGET
# ==========================

X = df.drop("fake", axis=1)
y = df["fake"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================
# TRAIN MODEL
# ==========================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================
# EVALUATE
# ==========================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

# ==========================
# SAVE MODEL (.pkl)
# ==========================

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("\n✅ model.pkl file saved successfully!")

# OPTIONAL (recommended)
# Save feature column order
pickle.dump(X.columns.tolist(), open("features.pkl", "wb"))

print("✅ features.pkl saved successfully!")
