print("--------------------- LINEAR REGRESSION ---------------------")

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

DATA_PATH = os.path.join(
    "src",
    "data",
    "dataset.csv"
)

FIGURES_PATH = os.path.join(
    "reports",
    "figures"
)

os.makedirs(FIGURES_PATH, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading Spotify dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])


# ============================================================
# SELECT FEATURES AND TARGET
# ============================================================

features = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms"
]

target = "popularity"


# ============================================================
# PREPARE DATA
# ============================================================

X = df[features].copy()
y = df[target].copy()

# Remove rows containing missing values
data = pd.concat([X, y], axis=1).dropna()

X = data[features]
y = data[target]

print("\nFeatures used:")
for feature in features:
    print("-", feature)

print("\nTarget variable:", target)


# ============================================================
# TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples :", len(X_train))
print("Testing samples  :", len(X_test))


# ============================================================
# FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# CREATE LINEAR REGRESSION MODEL
# ============================================================

model = LinearRegression()

model.fit(
    X_train_scaled,
    y_train
)


# ============================================================
# PREDICTIONS
# ============================================================

y_pred = model.predict(
    X_test_scaled
)


# ============================================================
# MODEL EVALUATION
# ============================================================

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = mse ** 0.5

mae = mean_absolute_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n" + "=" * 60)
print("LINEAR REGRESSION RESULTS")
print("=" * 60)

print(f"Mean Squared Error      : {mse:.4f}")
print(f"Root Mean Squared Error : {rmse:.4f}")
print(f"Mean Absolute Error     : {mae:.4f}")
print(f"R² Score                : {r2:.4f}")


# ============================================================
# MODEL COEFFICIENTS
# ============================================================

print("\n" + "=" * 60)
print("FEATURE COEFFICIENTS")
print("=" * 60)

coefficients = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
})

print(coefficients.to_string(index=False))


# ============================================================
# ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.4
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.title("Actual vs Predicted Spotify Popularity")
plt.xlabel("Actual Popularity")
plt.ylabel("Predicted Popularity")

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_PATH,
        "14_actual_vs_predicted_popularity.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# FEATURE COEFFICIENT GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.barh(
    coefficients["Feature"],
    coefficients["Coefficient"]
)

plt.title("Linear Regression Feature Coefficients")
plt.xlabel("Coefficient")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_PATH,
        "15_linear_regression_coefficients.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions = pd.DataFrame({
    "Actual_Popularity": y_test.values,
    "Predicted_Popularity": y_pred
})

predictions.to_csv(
    os.path.join(
        FIGURES_PATH,
        "linear_regression_predictions.csv"
    ),
    index=False
)


# ============================================================
# COMPLETION
# ============================================================

print("\nGraphs saved successfully to:")
print(FIGURES_PATH)

print("\n" + "=" * 60)
print("LINEAR REGRESSION COMPLETED")
print("=" * 60)