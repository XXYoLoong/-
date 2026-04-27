"""
Copyright 2026 Jiacheng Ni

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess(exercise_path: str, calories_path: str):
    """Load exercise/calorie files, merge them and build train/test features."""
    exercise_df = pd.read_excel(exercise_path)
    calories_df = pd.read_excel(calories_path)

    merged_df = exercise_df.merge(calories_df, on="User_ID", how="inner")
    merged_df["Gender_encoded"] = merged_df["Gender"].map({"female": 0, "male": 1})

    feature_cols = [
        "Gender_encoded",
        "Age",
        "Height",
        "Weight",
        "Duration",
        "Heart_Rate",
        "Body_Temp",
    ]

    X = merged_df[feature_cols]
    y = merged_df["Calories"]

    return train_test_split(X, y, test_size=0.2, random_state=42), feature_cols, merged_df


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train three regression models and return metrics plus predictions."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
    }

    rows = []
    predictions = {}

    for name, model in models.items():
        if name == "Random Forest Regressor":
            model.fit(X_train, y_train)
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
        else:
            model.fit(X_train_scaled, y_train)
            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)

        predictions[name] = test_pred
        rows.append(
            {
                "Model": name,
                "Train RMSE": np.sqrt(mean_squared_error(y_train, train_pred)),
                "Test RMSE": np.sqrt(mean_squared_error(y_test, test_pred)),
                "Train MAE": mean_absolute_error(y_train, train_pred),
                "Test MAE": mean_absolute_error(y_test, test_pred),
                "Train R^2": r2_score(y_train, train_pred),
                "Test R^2": r2_score(y_test, test_pred),
            }
        )

    return pd.DataFrame(rows), models, predictions


def create_scatter_plot(y_true, y_pred, title: str, filename: str):
    """Create an actual-vs-predicted scatter plot without requiring matplotlib."""
    width, height, margin = 800, 600, 80
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    range_val = max_val - min_val if max_val != min_val else 1.0

    draw.line((margin, height - margin, width - margin, height - margin), fill="black")
    draw.line((margin, height - margin, margin, margin), fill="black")
    draw.text((width / 2 - 120, 15), title, fill="black", font=font)
    draw.text((width / 2 - 30, height - margin + 30), "Actual", fill="black", font=font)
    draw.text((10, height / 2 - 30), "Predicted", fill="black", font=font)

    for actual, pred in zip(y_true, y_pred):
        x = margin + ((actual - min_val) / range_val) * (width - 2 * margin)
        y = height - margin - ((pred - min_val) / range_val) * (height - 2 * margin)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="blue")

    img.save(filename)


def create_bar_chart(labels, values, title: str, filename: str, bar_color="green"):
    """Create a simple bar chart without requiring matplotlib."""
    margin, bar_width, bar_spacing = 80, 60, 40
    width = margin * 2 + len(values) * bar_width + (len(values) - 1) * bar_spacing
    height = 600
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    max_val = max(values) * 1.1 if max(values) else 1.0
    draw.text((width / 2 - 110, 15), title, fill="black", font=font)

    for i, (label, val) in enumerate(zip(labels, values)):
        x0 = margin + i * (bar_width + bar_spacing)
        x1 = x0 + bar_width
        bar_h = (val / max_val) * (height - 2 * margin)
        y0 = height - margin - bar_h
        y1 = height - margin
        draw.rectangle([x0, y0, x1, y1], fill=bar_color)
        draw.text((x0, y0 - 15), f"{val:.2f}", fill="black", font=font)
        draw.text((x0, y1 + 5), str(label), fill="black", font=font)

    img.save(filename)


def main():
    parser = argparse.ArgumentParser(description="Calorie expenditure prediction project")
    parser.add_argument("--exercise", default="data/Exercise.csv.xlsx", help="Path to Exercise.csv.xlsx")
    parser.add_argument("--calories", default="data/Calories.csv.xlsx", help="Path to Calories.csv.xlsx")
    parser.add_argument("--output", default="results", help="Directory for result tables")
    parser.add_argument("--figures", default="figures", help="Directory for output figures")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.figures, exist_ok=True)

    (X_train, X_test, y_train, y_test), feature_cols, _ = load_and_preprocess(
        args.exercise,
        args.calories,
    )
    results_df, models, predictions = train_and_evaluate(X_train, X_test, y_train, y_test)

    result_file = os.path.join(args.output, "model_results.xlsx")
    results_df.to_excel(result_file, index=False)
    print(results_df.to_string(index=False))
    print(f"\nSaved result table: {result_file}")

    best_model_name = results_df.sort_values("Test RMSE").iloc[0]["Model"]
    best_pred = predictions[best_model_name]

    create_scatter_plot(
        list(y_test),
        list(best_pred),
        f"Actual vs Predicted Calories ({best_model_name})",
        os.path.join(args.figures, "predicted_vs_actual.png"),
    )

    create_bar_chart(
        list(results_df["Model"]),
        list(results_df["Test RMSE"]),
        "Model Comparison: Test RMSE",
        os.path.join(args.figures, "model_comparison.png"),
    )

    rf_importance = models["Random Forest Regressor"].feature_importances_
    sorted_idx = np.argsort(rf_importance)[::-1]
    create_bar_chart(
        [feature_cols[i] for i in sorted_idx],
        [rf_importance[i] for i in sorted_idx],
        "Random Forest Feature Importances",
        os.path.join(args.figures, "feature_importance.png"),
        bar_color="orange",
    )

    print(f"Best model: {best_model_name}")


if __name__ == "__main__":
    main()
