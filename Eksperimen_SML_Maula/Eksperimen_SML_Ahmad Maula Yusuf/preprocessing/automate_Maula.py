import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def preprocess_data(
    input_path="heart_raw/heart.csv",
    output_path="preprocessing/heart_preprocessing/heart_preprocessing.csv"
):
    df = pd.read_csv(input_path)

    # Drop duplicate data
    df = df.drop_duplicates()

    # Split features & target
    X = df.drop("target", axis=1)
    y = df["target"]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Combine processed data
    df_processed = pd.DataFrame(X_scaled, columns=X.columns)
    df_processed["target"] = y.values

    # Save processed dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_processed.to_csv(output_path, index=False)

if __name__ == "__main__":
    preprocess_data()
