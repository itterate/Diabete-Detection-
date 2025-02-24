from preprocess import load_and_preprocess_data
from train import train_models
from evaluate import evaluate_models
import config

if __name__ == "__main__":
    # Load & preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(config.DATA_PATH)

    # Train models
    models = train_models(X_train, X_test, y_train, y_test)

    # Evaluate models
    evaluate_models(models, X_test, y_test)
