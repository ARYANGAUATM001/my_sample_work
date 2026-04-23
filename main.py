# This is a sample Python script.
import nonlinear_benchmarks
import matplotlib.pyplot as plt

from models.gru_model import train_gru, simulate_gru
from models.deep_gru_model import train_deep_gru, simulate_deep_gru
from nonlinear_benchmarks.error_metrics import RMSE


# Load data
train, test = nonlinear_benchmarks.Cascaded_Tanks()
n = test.state_initialization_window_length


# Train models
model_gru = train_gru(train)
model_deep = train_deep_gru(train)


# Simulate
y_gru = simulate_gru(model_gru, test.u, list(test.y[:5]))
y_deep = simulate_deep_gru(model_deep, test.u, list(test.y[:5]))

y_gru = y_gru[:len(test.y)]
y_deep = y_deep[:len(test.y)]


# RMSE
rmse_gru = RMSE(test.y[n:], y_gru[n:])
rmse_deep = RMSE(test.y[n:], y_deep[n:])

print(f"GRU RMSE: {rmse_gru:.3f}")
print(f"Deep GRU RMSE: {rmse_deep:.3f}")


# Plot
plt.figure(figsize=(10, 4))

plt.plot(test_y, label="True Output")
plt.plot(y_gru, label="GRU")
plt.plot(y_deep, label="Deep GRU")

plt.xlabel("Time (steps)")      
plt.ylabel("Output value")        

plt.legend()
plt.title("Mamba-inspired Models Comparison")
plt.grid(True)                   

plt.savefig("results/model_comparison.png", dpi=300)
plt.show()                       





