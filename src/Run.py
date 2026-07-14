import pickle
import json
import os
import yaml
from evaluation import run_experiment
from model_train import model

if __name__ == "__main__":

    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("Evaluating...")
    run_experiment(config)

    # Save model
    os.makedirs("models", exist_ok=True)
    with open("models/churn_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved to models/churn_model.pkl")

    # Save metrics
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/results.json", "w") as f:
        json.dump(model.metrics, f, indent=2)
    print("Metrics saved to metrics/results.json")