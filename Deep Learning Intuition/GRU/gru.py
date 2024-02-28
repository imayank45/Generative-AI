import numpy as np

class GRU:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Parameters for update gate
        self.W_z = np.random.randn(hidden_size, input_size)
        self.U_z = np.random.randn(hidden_size, hidden_size)
        self.b_z = np.zeros((hidden_size, 1))
        
        # Parameters for reset gate
        self.W_r = np.random.randn(hidden_size, input_size)
        self.U_r = np.random.randn(hidden_size, hidden_size)
        self.b_r = np.zeros((hidden_size, 1))
        
        # Parameters for candidate activation
        self.W_h = np.random.randn(hidden_size, input_size)
        self.U_h = np.random.randn(hidden_size, hidden_size)
        self.b_h = np.zeros((hidden_size, 1))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def tanh(self, x):
        return np.tanh(x)
    
    def forward(self, x, h_prev):
        # Update gate
        z = self.sigmoid(np.dot(self.W_z, x) + np.dot(self.U_z, h_prev) + self.b_z)
        
        # Reset gate
        r = self.sigmoid(np.dot(self.W_r, x) + np.dot(self.U_r, h_prev) + self.b_r)
        
        # Candidate activation
        h_tilde = self.tanh(np.dot(self.W_h, x) + np.dot(self.U_h, np.multiply(r, h_prev)) + self.b_h)
        
        # New hidden state
        h_next = np.multiply(z, h_prev) + np.multiply(1 - z, h_tilde)
        
        return h_next

# Example usage
input_size = 10
hidden_size = 5

gru_cell = GRU(input_size, hidden_size)

# Input sequence
x = np.random.randn(input_size, 1)

# Initial hidden state
h_prev = np.zeros((hidden_size, 1))

# Forward pass through the GRU cell
h_next = gru_cell.forward(x, h_prev)

print("Input size:", input_size)
print("Hidden size:", hidden_size)
print("Input sequence x:", x.flatten())
print("Initial hidden state h_prev:", h_prev.flatten())
print("Output hidden state h_next:", h_next.flatten())
