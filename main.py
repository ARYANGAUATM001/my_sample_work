import nonlinear_benchmarks
import matplotlib.pyplot as plt

from model.gru_model import train_gru, simulate_gru
from model.deep_gru_model import train_deep_gru, simulate_deep_gru
from nonlinear_benchmarks.error_metrics import RMSE


train, test = nonlinear_benchmarks.Cascaded_Tanks()
n = test.state_initialization_window_length


model_gru = train_gru(train)
model_deep = train_deep_gru(train)


y_gru = simulate_gru(model_gru, test.u, list(test.y[:5]))
y_deep = simulate_deep_gru(model_deep, test.u, list(test.y[:5]))

y_gru = y_gru[:len(test.y)]
y_deep = y_deep[:len(test.y)]


rmse_gru = RMSE(test.y[n:], y_gru[n:])
rmse_deep = RMSE(test.y[n:], y_deep[n:])

print(f"GRU RMSE: {rmse_gru:.3f}")
print(f"Deep GRU RMSE: {rmse_deep:.3f}")


plt.figure(figsize=(10, 4))

true = test.y[:, 0] if len(test.y.shape) > 1 else test.y

plt.plot(true, label="True Output")
plt.plot(y_gru, label="GRU")
plt.plot(y_deep, label="Deep GRU")

plt.xlabel("Time Step")
plt.ylabel("Value")

plt.legend()
plt.grid(True)

plt.show()
