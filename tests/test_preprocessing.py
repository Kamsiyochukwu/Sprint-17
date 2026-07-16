from unittest import result
import pandas as pd
import numpy as np
import pytest
import sys
import yaml
from sklearn.model_selection import train_test_split

sys.path.insert(0, "src")
from data_preprocessing import validate_dataframe, clean_data, encode_categoricals, check_data_quality
from model_train import build_model

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

@pytest.fixture
def sample_data():
    """Small dataset that mimics the student dropout data structure."""
    return pd.DataFrame({
        "Age": [22.0, 20.0, np.nan, 24.0, 19.0, 21.0, np.nan],
        "Family_Income": [20000.0, 26000.0, np.nan, 50000.0, 30000.0, np.nan, 35000.0],
        "Study_Hours_per_Day": [3.5, 4.0, 2.0, 5.0, 1.5, 3.0, 4.5],
        "Attendance_Rate": [86.0, 92.0, 75.0, 88.0, 65.0, 95.0, 80.0],
        "Gender": ["Male", "Female", "Female", "Male", "Female", "Male", "Female"],
        "Internet_Access": ["Yes", "No", "Yes", "Yes", "No", "Yes", "No"],
        "Dropout": [1, 0, 1, 0, 1, 0, 0]
    })

class TestValidateDataframe:

    def test_valid_dataframe_passes(self, sample_data):
        result = validate_dataframe(
            sample_data,
            required_columns=["Age", "Gender", "Dropout"],
            target_column="Dropout"
        )
        assert result is True

    def test_missing_column_raises(self, sample_data):
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_dataframe(
                sample_data,
                required_columns=["Age", "Nonexistent"],
                target_column="Dropout"
            )

    def test_missing_target_raises(self, sample_data):
        with pytest.raises(ValueError, match="Target column"):
            validate_dataframe(
                sample_data,
                required_columns=["Age"],
                target_column="NonexistentTarget"
            )

    def test_empty_dataframe_raises(self):
        empty_df = pd.DataFrame({"Age": [], "Dropout": []})
        with pytest.raises(ValueError, match="empty"):
            validate_dataframe(empty_df, ["Age"], "Dropout")

class TestCleanData:

    def test_fills_numeric_nulls(self, sample_data):
        result = clean_data(sample_data, ["Age", "Family_Income"], [])
        assert result["Age"].isna().sum() == 0
        assert result["Family_Income"].isna().sum() == 0

    def test_does_not_modify_original(self, sample_data):
        original_nulls = sample_data["Age"].isna().sum()
        clean_data(sample_data, ["Age"], [])
        assert sample_data["Age"].isna().sum() == original_nulls

    def test_fills_with_median(self, sample_data):
        result = clean_data(sample_data, ["Age"], [])
        # Median of [22.0, 20.0, np.nan, 24.0, 19.0, 21.0, np.nan] = 21.0
        assert result["Age"].iloc[2] == 21.0

    def test_non_null_values_unchanged(self, sample_data):
        result = clean_data(sample_data, ["Age"], [])
        assert result["Age"].iloc[0] == 22.0
        assert result["Age"].iloc[1] == 20.0

class TestEncodeCategoricals:

    def test_creates_dummy_columns(self, sample_data):
        result = encode_categoricals(sample_data, ["Gender"])
        assert "Gender" not in result.columns
        assert any("Gender" in col for col in result.columns)

    def test_drops_first_category(self, sample_data):
        result = encode_categoricals(sample_data, ["Gender"])
        gender_cols = [col for col in result.columns if "Gender" in col]
        # drop_first=True means we should have 1 column, not 2
        assert len(gender_cols) == 1

    def test_preserves_row_count(self, sample_data):
        result = encode_categoricals(sample_data, ["Gender", "Internet_Access"])
        assert len(result) == len(sample_data)

class TestDataQuality:

    def test_counts_nulls(self, sample_data):
        report = check_data_quality(sample_data, ["Age", "Family_Income"])
        assert report["total_nulls"] == 4  # one in Age, one in Family_Income

    def test_counts_rows(self, sample_data):
        report = check_data_quality(sample_data, ["Age"])
        assert report["total_rows"] == 7

    def test_reports_numeric_ranges(self, sample_data):
        report = check_data_quality(sample_data, ["Attendance_Rate"])
        assert report["Attendance_Rate_min"] == 65.0
        assert report["Attendance_Rate_max"] == 95.0

class TestDataValidation:

    def test_expected_columns_present(self):
        df = pd.read_csv(config["data_url"])
        expected_cols = ["Age", "Attendance_Rate", "Study_Hours_per_Day", "Dropout"]
        for col in expected_cols:
            assert col in df.columns

    def test_target_variable_valid_values(self):
        df = pd.read_csv(config["data_url"])
        valid_values = {0, 1}
        actual_values = set(df["Dropout"].unique())
        assert actual_values.issubset(valid_values)

    def test_numeric_features_within_range(self):
        df = pd.read_csv(config["data_url"])
        assert df["Attendance_Rate"].between(0, 100).all()
        assert df["Age"].between(15, 60).all()

class TestModelValidation:

    def test_model_predictions_shape_and_type(self, sample_data):
        X = sample_data.drop(columns=["Dropout", "Gender", "Internet_Access"])  # Drop categorical for simplicity
        y = sample_data["Dropout"]

        X = clean_data(X, ["Age", "Family_Income"], [])

        for test_size in config["test_size"]:
            X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=config["random_state"],
            stratify=y
            )
            model = build_model(config)
            model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        assert predictions.shape[0] == y_test.shape[0]
        assert isinstance(predictions, np.ndarray)

    def test_model_accuracy_threshold(self, sample_data):
        X = sample_data.drop(columns=["Dropout", "Gender", "Internet_Access"])  # Drop categorical for simplicity
        y = sample_data["Dropout"]

        X = clean_data(X, ["Age", "Family_Income"], [])

        for test_size in config["test_size"]:
            X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=config["random_state"],
            stratify=y
            )
            model = build_model(config)
            model.fit(X_train, y_train)

            accuracy = model.score(X_test, y_test)
            assert accuracy >= 0.5

    