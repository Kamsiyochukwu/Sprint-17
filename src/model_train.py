import pandas as pd
import numpy as np
import sys
import mlflow
import mlflow.sklearn
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import yaml

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

sys.path.insert(0, os.path.dirname(__file__))
from data_preprocessing import validate_dataframe, clean_data, encode_categoricals, check_data_quality



def load_data(url):
    df = pd.read_csv(url)

    return df

def load_and_prepare_data(config):
    """Load the student dropout dataset and prepare it for training."""

    df = load_data(config["data_url"])

    validate_dataframe(df, config['numeric_columns'] + config['categorical_columns'], config['target'])
    check_data_quality(df, config["numeric_columns"])
    df = clean_data(df, config['numeric_columns'], config['categorical_columns'])
    df = encode_categoricals(df, config['categorical_columns'])

    # Separate features and target
    X = df.drop(columns=["Dropout"])
    y = df["Dropout"]
    n_rows = len(df)

    return X, y, n_rows

def build_model(config):
    """Build a model based on the config."""
    if config["model_type"] == "logistic_regression":
        return LogisticRegression(
            C=config["lr_C"],
            random_state=config["random_state"],
            max_iter=1000
        )
    elif config["model_type"] == "random_forest":
        return RandomForestClassifier(
            n_estimators=config["rf_n_estimators"],
            max_depth=config["rf_max_depth"],
            random_state=config["random_state"]
        )
    elif config["model_type"] == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=config["gb_n_estimators"],
            learning_rate=config["gb_learning_rate"],
            max_depth=config["gb_max_depth"],
            random_state=config["random_state"]
        )
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")
    
def model(config):
    """Train a model based on the config and return it."""
    # Load and prepare data ──
    X, y, n_rows = load_and_prepare_data(config)

    mlflow.log_param("n_rows", n_rows)
    mlflow.log_param("n_features", X.shape[1])

    # ── Split data ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y
    )

    # ── Optionally scale features ──
    if config["scale_features"]:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # ── Train ──
    model = build_model(config)
    print(f"\nTraining {config['model_type']}...")
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

if __name__ == "__main__":
    print("Training model...")
    model = model(config)