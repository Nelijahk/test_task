import pandas as pd
from sklearn import linear_model, svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn, optim


class RegressionModel(nn.Module):
    def __init__(self, input_size):
        super(RegressionModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)


df = pd.read_csv("train.csv")
df_test = pd.read_csv("hidden_test.csv")
scaler = MinMaxScaler()
X = df.drop('target', axis=1)
y = df['target']
X_scaled = scaler.fit_transform(X)
# print(df)
# print(df_scaled)
# print(df.isna().sum())
# print(df_test)
# print(df_test.info())

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=666)

# model = linear_model.LinearRegression().fit(X_train, y_train)
# model = linear_model.Lasso().fit(X_train, y_train)
# model = RandomForestRegressor(max_depth=4).fit(X_train, y_train)
# y_pred = model.predict(X_test)
# # print(reg.score(X, y))
# print(root_mean_squared_error(y_test, y_pred))

y_train_np = y_train.values.reshape(-1, 1)
y_test_np = y_test.values.reshape(-1, 1)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test_np, dtype=torch.float32)

# Ініціалізація моделі
model = RegressionModel(X_train.shape[1])

# Функція втрат для регресії (MSE)
criterion = nn.MSELoss()

# Оптимізатор (Adam є гарним вибором за замовчуванням)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Гіперпараметри навчання
epochs = 500
batch_size = 64
N = X_train.shape[0]

# --- Цикл навчання ---
for epoch in range(epochs):
    # Встановлення режиму навчання
    model.train()

    # Випадковий перемішування даних (для простоти використовуємо повний батч)
    # Якщо хочете батчі, використовуйте DataLoader

    # 1. Прямий прохід (Forward Pass)
    y_pred = model(X_train)

    # 2. Обчислення втрат
    loss = criterion(y_pred, y_train)

    # 3. Зворотний прохід (Backward Pass)
    optimizer.zero_grad()  # Обнулення градієнтів з попередньої ітерації
    loss.backward()  # Обчислення градієнтів

    # 4. Оновлення ваг
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}')

model.eval()

# Відключаємо обчислення градієнтів, оскільки ми не навчаємо
with torch.no_grad():
    y_pred_tensor = model(X_test)

    # Перетворення результатів назад у numpy для метрики scikit-learn
    y_test_np = y_test.numpy()
    y_pred_np = y_pred_tensor.numpy()

    # Обчислення RMSE на оригінальних (немасштабованих) значеннях
    rmse_value = root_mean_squared_error(y_test_np, y_pred_np)

    print("\n--- Результат оцінки ---")
    print(f"RMSE : {rmse_value:.4f}")