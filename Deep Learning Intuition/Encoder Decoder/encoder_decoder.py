import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

# Assuming encoder_hidden and decoder_hidden are torch tensors of shape (num_layers, batch_size, hidden_size)
encoder_hidden = torch.randn(1, 1, 3)  # Example encoder hidden state
decoder_hidden = torch.randn(1, 1, 3)  # Example decoder hidden state

# Visualize Encoder Hidden State
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(encoder_hidden[0, 0, 0], encoder_hidden[0, 0, 1], encoder_hidden[0, 0, 2], c='r', marker='o')
ax1.set_title('Encoder Hidden State')
ax1.set_xlabel('Dimension 1')
ax1.set_ylabel('Dimension 2')
ax1.set_zlabel('Dimension 3')

# Visualize Decoder Hidden State
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(decoder_hidden[0, 0, 0], decoder_hidden[0, 0, 1], decoder_hidden[0, 0, 2], c='b', marker='o')
ax2.set_title('Decoder Hidden State')
ax2.set_xlabel('Dimension 1')
ax2.set_ylabel('Dimension 2')
ax2.set_zlabel('Dimension 3')

plt.tight_layout()
plt.show()
