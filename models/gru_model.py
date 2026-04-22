import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def build_sequences(u, y, seq_len=5):
    X, Y = [], []
    for i in range(seq_len, len(y)):
        seq = [[u[j], y[j]] for j in range(i - seq_len, i)]
        X.append(seq)
        Y.append(y[i])
    return np.array(X), np.array(Y)


class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(2, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


def train_gru(data):
    X, y = build_sequences(data.u, data.y)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    model = GRUModel()
    opt = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for _ in range(50):
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()

    return model


def simulate_gru(model, u, y_init, seq_len=5):
    model.eval()
    y_pred = list(y_init)

    for t in range(seq_len, len(u)):
        seq = [[u[j], y_pred[j]] for j in range(t - seq_len, t)]
        x = torch.tensor([seq], dtype=torch.float32)

        with torch.no_grad():
            y_next = model(x).item()

        y_next = np.clip(y_next, 0, 10)
        y_pred.append(y_next)

    return np.array(y_pred)