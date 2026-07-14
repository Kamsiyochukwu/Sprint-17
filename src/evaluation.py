import pandas as pd
import numpy as np
import os
import sys
import pickle
import json
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import yaml

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

sys.path.insert(0, os.path.dirname(__file__))
from model_train import model

def run_experiment(config):
    """Run a single experiment with the given config, tracked by MLflow."""

    # Set the experiment name so all runs are grouped together
    mlflow.set_experiment("student-dropout-prediction")

    # Start an MLflow run
    with mlflow.start_run():

        modelplk, X_test, y_test = model(config)

        # ── Log all configuration as parameters ──
        mlflow.log_param("model_type", config["model_type"])
        mlflow.log_param("test_size", config["test_size"])
        mlflow.log_param("random_state", config["random_state"])
        mlflow.log_param("handle_missing", config["handle_missing"])
        mlflow.log_param("scale_features", config["scale_features"])
        mlflow.log_param("features_dropped", str(config["features_to_drop"]))

        # Log model-specific hyperparameters based on model type
        if config["model_type"] == "logistic_regression":
            mlflow.log_param("C", config["lr_C"])
        elif config["model_type"] == "random_forest":
            mlflow.log_param("n_estimators", config["rf_n_estimators"])
            mlflow.log_param("max_depth", str(config["rf_max_depth"]))
        elif config["model_type"] == "gradient_boosting":
            mlflow.log_param("n_estimators", config["gb_n_estimators"])
            mlflow.log_param("learning_rate", config["gb_learning_rate"])
            mlflow.log_param("max_depth", config["gb_max_depth"])

        # ── Evaluate ──
        y_pred = modelplk.predict(X_test)
        y_prob = modelplk.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc_roc": auc
        }

        # ── Log metrics ──
        mlflow.log_metric("accuracy", round(accuracy, 4))
        mlflow.log_metric("precision", round(precision, 4))
        mlflow.log_metric("recall", round(recall, 4))
        mlflow.log_metric("f1_score", round(f1, 4))
        mlflow.log_metric("auc_roc", round(auc, 4))

        # ── Log the trained model as an artifact ──
        mlflow.sklearn.log_model(modelplk, "model")

        # ── Log the config file as an artifact for reference ──
        config_path = "config_snapshot.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)
        mlflow.log_artifact(config_path)
        os.remove(config_path)  # clean up temp file

        # ── Print results ──
        print(f"\n{'='*50}")
        print(f"Model:     {config['model_type']}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"AUC-ROC:   {auc:.4f}")
        print(f"{'='*50}")

        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow Run ID: {run_id}")
        print("View this run in the UI: mlflow ui")

    return run_id, metrics, modelplk