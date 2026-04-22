import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.gru_model import build_sequences


class DeepGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(2, 64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


def train_deep_gru(data):
    X, y = build_sequences(data.u, data.y)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    model = DeepGRU()
    opt = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for _ in range(60):
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()

    return model


def simulate_deep_gru(model, u, y_init, seq_len=5):
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