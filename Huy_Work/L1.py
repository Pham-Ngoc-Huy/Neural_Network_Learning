import pandas as pd
import numpy as np

df = pd.read_csv('/root/Neural_Network/DL_Tutorial/L1/data_linear.csv')
x = df['Square'].values
y = df['Price'].values
N = len(x)

# 2. Initialize parameters
w0 = 0.0
w1 = 0.0
learning_rate = 0.0001
epochs = 1000

# 3. Gradient Descent Loop
for epoch in range(epochs):
    y_pred = w0 * x + w1
    error = y_pred - y
    
    # MSE Loss (optional to track)
    loss = (1/(2*N)) * np.sum(error**2)
    
    # Gradients
    dw0 = (1/N) * np.dot(error, x)
    dw1 = (1/N) * np.sum(error)

    # Update parameters
    w0 -= learning_rate * dw0
    w1 -= learning_rate * dw1

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}, w = {w0:.4f}, b = {w1:.4f}")
        print(f'MSE: {loss}')
# 4. Final parameters
print(f"Final model: y = {w0:.2f} * x + {w1:.2f}")
# 5. Prediction
x1 = 50
y1 = w0 + x1 * w1
print(y1)
