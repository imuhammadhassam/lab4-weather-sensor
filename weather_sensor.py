import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import zscore

data = {
    "Hour": list(range(1, 16)),
    "Temperature": [
        22, 23, np.nan, np.nan, np.nan,
        25, 26, 58, 24, 23,
        22, 21, 23, 24, 25
    ]
}

df = pd.DataFrame(data)

df["Temp_raw"] = df["Temperature"]

# Statistics before cleaning
before_clean = df["Temperature"].dropna()

print("===== STATS BEFORE CLEANING =====")
print("Mean:", before_clean.mean())
print("Std:", before_clean.std())
print("Skewness:", stats.skew(before_clean))
print("Kurtosis:", stats.kurtosis(before_clean))

# Linear interpolation
df["Temp_clean"] = df["Temperature"].interpolate(method="linear")

print("\nAfter interpolation, missing values:",
      df["Temp_clean"].isnull().sum())

# Detect outlier using Z-score threshold 2
temp_z = zscore(df["Temp_clean"])
df["Detection_Zscore"] = temp_z

outlier_mask = df["Detection_Zscore"].abs() > 2

if outlier_mask.any():
    outlier_rows = df[outlier_mask]

    print("\nOutlier detected:")
    for _, row in outlier_rows.iterrows():
        print(
            "Hour:",
            int(row["Hour"]),
            "Value:",
            row["Temp_clean"]
        )

    # Median calculated from interpolated data
    median_temp = df["Temp_clean"].median()

    print("Replacing outlier with median:", median_temp)

    df.loc[outlier_mask, "Temp_clean"] = median_temp

# Final standardization
df["Temp_Zscore"] = zscore(df["Temp_clean"])

print("\n===== BEFORE / AFTER COMPARISON =====")
print(
    df[
        ["Hour", "Temp_raw", "Temp_clean", "Temp_Zscore"]
    ]
)

print("\n===== STATS AFTER CLEANING =====")
print("Mean:", df["Temp_clean"].mean())
print("Std:", df["Temp_clean"].std())