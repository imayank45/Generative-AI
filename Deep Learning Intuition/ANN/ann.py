import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define layers
layers = ['Input', 'Hidden 1', 'Hidden 2', 'Output']
num_layers = len(layers)

# Define neuron counts for each layer
neuron_counts = [2, 100, 32, 1]

# Plot neurons for each layer
for layer_idx, layer_name in enumerate(layers):
    neuron_count = neuron_counts[layer_idx]
    xs = [layer_idx] * neuron_count
    ys = range(neuron_count)
    zs = [0] * neuron_count  # All neurons in the same layer have the same z-coordinate
    ax.scatter(xs, ys, zs, label=f'{layer_name} ({neuron_count} neurons)', s=50)

# Plot connections between neurons
for layer_idx in range(num_layers - 1):
    neuron_count_current_layer = neuron_counts[layer_idx]
    neuron_count_next_layer = neuron_counts[layer_idx + 1]
    for i in range(neuron_count_current_layer):
        for j in range(neuron_count_next_layer):
            ax.plot([layer_idx, layer_idx + 1], [i, j], [0, 0], color='gray', alpha=0.5)

# Set labels and legend
ax.set_xlabel('Layer')
ax.set_ylabel('Neuron Index')
ax.set_zlabel('')
ax.set_title('Neural Network Architecture (3D)')
ax.legend()

# Show plot
plt.show()
