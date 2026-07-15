import pandas as pd
import numpy as np
import yaml
from evidently import Report
from evidently.presets import DataDriftPreset

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load the data
df = pd.read_csv(config["data_url"])

df = df.drop(columns=["Student_ID"])

def create_reference_and_production(df):
    """
    Split data into reference (training) and production batches.
    The reference set represents what the model was trained on.
    Production batches simulate data arriving over three months.
    """
    # First 60% is the reference (training) data
    split = int(len(df) * 0.6)
    reference = df.iloc[:split].copy()
    remaining = df.iloc[split:].copy()

    # Split remaining into three production "months"
    batch_size = len(remaining) // 3
    month1 = remaining.iloc[:batch_size].copy()
    month2 = remaining.iloc[batch_size:batch_size*2].copy()
    month3 = remaining.iloc[batch_size*2:].copy()

    return reference, month1, month2, month3

reference, month1, month2, month3 = create_reference_and_production(df)


def introduce_drift(month2, month3):
    """
    Simulate realistic drift in months 2 and 3.
    Month 1 stays clean to show what 'no drift' looks like.
    """
    # Month 2: moderate drift
    # Study hours increase (new tutoring program makes students study more)
    month2["Study_Hours_per_Day"] = month2["Study_Hours_per_Day"] + np.random.normal(1.0, 0.3, len(month2))
    month2["Study_Hours_per_Day"] = month2["Study_Hours_per_Day"].clip(0, 12)

    # Stress index increases slightly
    month2["Stress_Index"] = month2["Stress_Index"] + np.random.normal(0.5, 0.2, len(month2))
    month2["Stress_Index"] = month2["Stress_Index"].clip(0, 10)

    # Month 3: significant drift
    # Family income shifts (new scholarship attracts wealthier students)
    month3["Family_Income"] = month3["Family_Income"] * np.random.uniform(1.3, 1.8, len(month3))

    # Study hours shift even more
    month3["Study_Hours_per_Day"] = month3["Study_Hours_per_Day"] + np.random.normal(2.0, 0.5, len(month3))
    month3["Study_Hours_per_Day"] = month3["Study_Hours_per_Day"].clip(0, 12)

    # Age distribution changes (adult learner program)
    adult_learners = np.random.uniform(28, 45, int(len(month3) * 0.3))
    indices = np.random.choice(month3.index, size=len(adult_learners), replace=False)
    month3.loc[indices, "Age"] = adult_learners

    # Department distribution shifts
    dept_shift_indices = np.random.choice(month3.index, size=int(len(month3) * 0.2), replace=False)
    month3.loc[dept_shift_indices, "Department"] = "CS"

    # Attendance drops across the board
    month3["Attendance_Rate"] = month3["Attendance_Rate"] - np.random.normal(8, 3, len(month3))
    month3["Attendance_Rate"] = month3["Attendance_Rate"].clip(0, 100)

    return month2, month3

month2, month3 = introduce_drift(month2, month3)

print("=" * 60)
print("DRIFT REPORT: Month 1 (expected: no drift)")
print("=" * 60)

report_month1 = Report(metrics=[DataDriftPreset()])
snapshot_month1 = report_month1.run(reference_data=reference, current_data=month1)
snapshot_month1.save_html("reports/drift_month1.html")
print("Report saved to reports/drift_month1.html")

print("\n" + "=" * 60)
print("DRIFT REPORT: Month 2 (expected: moderate drift)")
print("=" * 60)

report_month2 = Report(metrics=[DataDriftPreset()])
snapshot_month2 = report_month2.run(reference_data=reference, current_data=month2)
snapshot_month2.save_html("reports/drift_month2.html")
print("Report saved to reports/drift_month2.html")

print("\n" + "=" * 60)
print("DRIFT REPORT: Month 3 (expected: significant drift)")
print("=" * 60)

report_month3 = Report(metrics=[DataDriftPreset()])
snapshot_month3 = report_month3.run(reference_data=reference, current_data=month3)
snapshot_month3.save_html("reports/drift_month3.html")
print("Report saved to reports/drift_month3.html")

print("\nOpen the HTML files in your browser to explore the reports.")