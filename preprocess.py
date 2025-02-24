import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import config

def load_and_preprocess_data(file_path):
    """Load dataset, preprocess features and split into train & test."""
    df = pd.read_csv(file_path)

    # Separate features and target
    X = df.drop(columns=["target"])  # Replace "target" with actual column name
    y = df["target"]

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    return X_train, X_test, y_train, y_test
