import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import os
import joblib

from RegressionClasses import RegressionModel, RegressionDataset
import config


def train(train_loader):
    reg_model = RegressionModel(X.shape[1])
    criterion = nn.MSELoss()
    optimiser = optim.Adam(reg_model.parameters(), lr=0.0001)

    print("\nTRAINING IS STARTED")
    epochs = config.EPOCHS
    for epoch in range(epochs):
        reg_model.train()
        for X_batch, y_batch in train_loader:
            y_pred = reg_model(X_batch)
            # now it is rmse loss
            loss = torch.sqrt(criterion(y_pred, y_batch))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss RMSE: {loss.item():.6f}')
    print("\nTRAINING IS FINISHED")

    PATH = config.MODEL_WEIGHTS_FILENAME
    try:
        torch.save(reg_model.state_dict(), PATH)
        print(f"\nModel is saved -> {os.path.abspath(PATH)}")
    except Exception as e:
        print(f"\nSaving ERROR: {e}")


if __name__ == "__main__":
    df = pd.read_csv(config.TRAIN_DATA_PATH)

    X = df.drop('target', axis=1)
    y = df['target']
    y = y.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, config.SCALER_FILENAME)

    train_dataset = RegressionDataset(X_scaled, y)

    batch_size = config.BATCH_SIZE
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        # after each epoch data is shuffled
        shuffle=True
    )

    train(train_loader)
