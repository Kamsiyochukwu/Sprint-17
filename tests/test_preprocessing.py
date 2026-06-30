import pandas as pd
import numpy as np
import pytest
import sys

sys.path.insert(0, "src")
from data_preprocessing import validate_dataframe, clean_data, encode_categoricals, check_data_quality

@pytest.fixture
def sample_data():
    """Small dataset that mimics the student dropout data structure."""
    return pd.DataFrame({
        "Age": [22.0, 20.0, np.nan, 24.0, 19.0, 21.0],
        "Family_Income": [25000.0, 40000.0, 35000.0, np.nan, 50000.0, 30000.0],
        "Study_Hours_per_Day": [3.5, 2.0, 4.0, 1.5, 5.0, 3.0],
        "Attendance_Rate": [85.0, 70.0, 90.0, 65.0, 95.0, 80.0],
        "Gender": ["Male", "Female", "Male", "Female", "Male", "Female"],
        "Internet_Access": ["Yes", "Yes", "No", "Yes", "No", "Yes"],
        "Dropout": [0, 1, 0, 1, 0, 0]
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
        # Median of [22, 20, 24, 19, 21] = 21
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
        assert report["total_nulls"] == 2  # one in Age, one in Family_Income

    def test_counts_rows(self, sample_data):
        report = check_data_quality(sample_data, ["Age"])
        assert report["total_rows"] == 6

    def test_reports_numeric_ranges(self, sample_data):
        report = check_data_quality(sample_data, ["Attendance_Rate"])
        assert report["Attendance_Rate_min"] == 65.0
        assert report["Attendance_Rate_max"] == 95.0