print("--------------------- GRADIENT DESCENT ---------------------")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


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
# FEATURES AND TARGET
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

data = df[features + [target]].dropna()

X = data[features].values
y = data[target].values


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

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ============================================================
# ADD INTERCEPT COLUMN
# ============================================================

X_train_gd = np.c_[
    np.ones(X_train.shape[0]),
    X_train
]

X_test_gd = np.c_[
    np.ones(X_test.shape[0]),
    X_test
]


# ============================================================
# GRADIENT DESCENT FUNCTION
# ============================================================

def gradient_descent(
    X,
    y,
    learning_rate,
    iterations
):

    n_samples = X.shape[0]
    n_features = X.shape[1]

    weights = np.zeros(n_features)

    cost_history = []

    for i in range(iterations):

        predictions = X @ weights

        errors = predictions - y

        gradients = (
            2 / n_samples
        ) * (
            X.T @ errors
        )

        weights = (
            weights
            - learning_rate * gradients
        )

        cost = np.mean(
            errors ** 2
        )

        cost_history.append(cost)

    return weights, cost_history


# ============================================================
# LEARNING RATES
# ============================================================

learning_rates = [
    0.001,
    0.01,
    0.05
]

iterations = 1000

results = []


# ============================================================
# TRAIN WITH DIFFERENT LEARNING RATES
# ============================================================

for learning_rate in learning_rates:

    print("\n" + "=" * 60)
    print(
        f"Training with learning rate: {learning_rate}"
    )
    print("=" * 60)

    weights, cost_history = gradient_descent(
        X_train_gd,
        y_train,
        learning_rate,
        iterations
    )

    predictions = (
        X_test_gd @ weights
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = mse ** 0.5

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print(
        f"MSE  : {mse:.4f}"
    )

    print(
        f"RMSE : {rmse:.4f}"
    )

    print(
        f"MAE  : {mae:.4f}"
    )

    print(
        f"R²   : {r2:.4f}"
    )

    results.append({
        "Learning_Rate": learning_rate,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    })


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("GRADIENT DESCENT RESULTS")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# COST VS ITERATIONS
# ============================================================

plt.figure(figsize=(10, 6))

for learning_rate in learning_rates:

    weights, cost_history = gradient_descent(
        X_train_gd,
        y_train,
        learning_rate,
        iterations
    )

    plt.plot(
        range(iterations),
        cost_history,
        label=f"Learning Rate = {learning_rate}"
    )

plt.title(
    "Gradient Descent Cost vs Iterations"
)

plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error")

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_PATH,
        "16_gradient_descent_cost.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# R² SCORE COMPARISON
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
    results_df["Learning_Rate"].astype(str),
    results_df["R2"]
)

plt.title(
    "Gradient Descent R² Score by Learning Rate"
)

plt.xlabel("Learning Rate")
plt.ylabel("R² Score")

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_PATH,
        "17_gradient_descent_r2_comparison.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# COMPARE WITH SCIKIT-LEARN
# ============================================================

sklearn_model = LinearRegression()

sklearn_model.fit(
    X_train,
    y_train
)

sklearn_predictions = sklearn_model.predict(
    X_test
)

sklearn_mse = mean_squared_error(
    y_test,
    sklearn_predictions
)

sklearn_rmse = sklearn_mse ** 0.5

sklearn_mae = mean_absolute_error(
    y_test,
    sklearn_predictions
)

sklearn_r2 = r2_score(
    y_test,
    sklearn_predictions
)


print("\n" + "=" * 60)
print("SCIKIT-LEARN LINEAR REGRESSION COMPARISON")
print("=" * 60)

print(
    f"MSE  : {sklearn_mse:.4f}"
)

print(
    f"RMSE : {sklearn_rmse:.4f}"
)

print(
    f"MAE  : {sklearn_mae:.4f}"
)

print(
    f"R²   : {sklearn_r2:.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    os.path.join(
        FIGURES_PATH,
        "gradient_descent_results.csv"
    ),
    index=False
)


# ============================================================
# COMPLETION
# ============================================================

print("\nGraphs saved successfully to:")
print(FIGURES_PATH)

print("\n" + "=" * 60)
print("GRADIENT DESCENT COMPLETED")
print("=" * 60)