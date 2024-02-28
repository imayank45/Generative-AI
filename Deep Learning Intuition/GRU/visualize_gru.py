import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define sigmoid and tanh functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

# Input
x = 0.5
h_prev = 0.2

# Parameters
W_z = 0.1
U_z = 0.2
b_z = 0.05

W_r = -0.1
U_r = 0.15
b_r = 0.03

W_h = 0.08
U_h = -0.25
b_h = 0.02

# Compute update gate
z = sigmoid(W_z * x + U_z * h_prev + b_z)

# Compute reset gate
r = sigmoid(W_r * x + U_r * h_prev + b_r)

# Compute candidate activation
h_tilde = tanh(W_h * x + U_h * (r * h_prev) + b_h)

# Compute output
h_next = (1 - z) * h_prev + z * h_tilde

# Visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot gates
ax.scatter(W_z * x + U_z * h_prev + b_z, 0, z, c='r', marker='o', label='Update Gate (z)')
ax.scatter(W_r * x + U_r * h_prev + b_r, 0, r, c='g', marker='o', label='Reset Gate (r)')
ax.scatter(W_h * x + U_h * (r * h_prev) + b_h, 0, h_tilde, c='b', marker='o', label='Candidate Activation (h_tilde)')
ax.scatter(0, 0, h_next, c='m', marker='o', label='Output (h_next)')

ax.set_xlabel('Weighted Input')
ax.set_ylabel('Weighted Hidden State')
ax.set_zlabel('Output')

ax.legend()
plt.title('3D Visualization of GRU Cell Operations')

plt.show()
