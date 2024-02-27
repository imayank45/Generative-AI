import numpy as np
import matplotlib.pyplot as plt

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights and biases
        self.Wf = np.random.randn(hidden_size + input_size, hidden_size)
        self.bf = np.zeros((1, hidden_size))
        
        self.Wi = np.random.randn(hidden_size + input_size, hidden_size)
        self.bi = np.zeros((1, hidden_size))
        
        self.Wc = np.random.randn(hidden_size + input_size, hidden_size)
        self.bc = np.zeros((1, hidden_size))
        
        self.Wo = np.random.randn(hidden_size + input_size, hidden_size)
        self.bo = np.zeros((1, hidden_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def tanh(self, x):
        return np.tanh(x)
    
    def forward(self, xt, ht_1, Ct_1):
        # Concatenate input and previous hidden state
        concat_input = np.concatenate((ht_1, xt), axis=1)
        
        # Forget gate
        ft = self.sigmoid(np.dot(concat_input, self.Wf) + self.bf)
        
        # Input gate and candidate values
        it = self.sigmoid(np.dot(concat_input, self.Wi) + self.bi)
        C_tilde = self.tanh(np.dot(concat_input, self.Wc) + self.bc)
        
        # Update cell state
        Ct = ft * Ct_1 + it * C_tilde
        
        # Output gate
        ot = self.sigmoid(np.dot(concat_input, self.Wo) + self.bo)
        
        # Compute new hidden state
        ht = ot * self.tanh(Ct)
        
        return ht, Ct, ft, it, ot

# Create an LSTM cell instance
input_size = 10
hidden_size = 5
lstm_cell = LSTMCell(input_size, hidden_size)

# Define input
xt = np.random.randn(1, input_size)
ht_1 = np.random.randn(1, hidden_size)
Ct_1 = np.random.randn(1, hidden_size)

# Forward pass through LSTM cell
ht, Ct, ft, it, ot = lstm_cell.forward(xt, ht_1, Ct_1)

# Visualize LSTM operations
plt.figure(figsize=(10, 6))

# Plot forget gate
plt.subplot(2, 2, 1)
plt.bar(range(hidden_size), ft.flatten(), color='blue')
plt.title('Forget Gate')
plt.xlabel('Hidden Unit')
plt.ylabel('Activation')

# Plot input gate
plt.subplot(2, 2, 2)
plt.bar(range(hidden_size), it.flatten(), color='green')
plt.title('Input Gate')
plt.xlabel('Hidden Unit')
plt.ylabel('Activation')

# Plot output gate
plt.subplot(2, 2, 3)
plt.bar(range(hidden_size), ot.flatten(), color='red')
plt.title('Output Gate')
plt.xlabel('Hidden Unit')
plt.ylabel('Activation')

# Plot cell state
plt.subplot(2, 2, 4)
plt.bar(range(hidden_size), Ct.flatten(), color='purple')
plt.title('Cell State')
plt.xlabel('Hidden Unit')
plt.ylabel('Value')

plt.tight_layout()
plt.show()
