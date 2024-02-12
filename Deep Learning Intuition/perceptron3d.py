import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def feed_forward(X, W1, b1, W2, b2):
    hidden_layer_input = np.dot(X, W1) + b1
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, W2) + b2
    output = sigmoid(output_layer_input)
    return output

# Generate some random data
np.random.seed(0)
X = np.random.rand(100, 2)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

# Initialize weights and biases
W1 = np.random.randn(2, 3)
b1 = np.random.randn(1, 3)
W2 = np.random.randn(3, 1)
b2 = np.random.randn(1, 1)

# Plot the data points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, c=y, cmap='viridis')

# Plot the decision boundary
x1_grid, x2_grid = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
X_grid = np.column_stack([x1_grid.flatten(), x2_grid.flatten()])
y_grid = feed_forward(X_grid, W1, b1, W2, b2).reshape(x1_grid.shape)
ax.plot_surface(x1_grid, x2_grid, y_grid, alpha=0.5, cmap='viridis')

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Prediction')
ax.set_title('Decision Boundary for a Neural Network')

plt.show()
