import pandas as pd
import joblib
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

import config
from RegressionClasses import RegressionModel


def predict(model, data_loader):
    print("\nPREDICTION IS STARTED")
    predictions = []
    with torch.no_grad():
        for X_batch in data_loader:
            X_batch = X_batch[0]
            y_pred_tensor = model(X_batch)
            predictions.append(y_pred_tensor.numpy())
    y_pred_final = np.concatenate(predictions).flatten()
    print("\nPREDICTION IS FINISHED")
    return y_pred_final


def save_results(predictions):
    results = pd.DataFrame(data=predictions, columns=['target'])
    results.to_csv(config.OUTPUT_PREDICTIONS_PATH, index=False)
    print(f"\nResults saved into -> {os.path.abspath(config.OUTPUT_PREDICTIONS_PATH)}")


if __name__ == "__main__":
    df = pd.read_csv(config.TEST_DATA_PATH)

    scaler = joblib.load(config.SCALER_FILENAME)
    X_scaled = scaler.fit_transform(df)

    X_test_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    test_dataset = TensorDataset(X_test_tensor)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    reg_model = RegressionModel(X_scaled.shape[1])
    reg_model.load_state_dict(torch.load(config.MODEL_WEIGHTS_FILENAME))
    reg_model.eval()
    print(f"\nModel '{config.MODEL_WEIGHTS_FILENAME}' is loaded successfully")

    predictions = predict(reg_model, test_loader)
    save_results(predictions)
