import numpy as np

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
        
        return ht, Ct

# Function to generate data for next word prediction task
def generate_data():
    # Sample vocabulary
    vocabulary = ["after", "the", "initializing", "LSTM", "cell", "with", "a", "given", "input", "size", "and", "hidden", "we", "performed", "forward", "pass", "through", "updating", "hidden", "state", "based", "on", "at", "time", "step", "previous", "and", "cell", "performance", "by"]
    
    # Generate random sentence
    sentence_length = np.random.randint(5, 15)
    sentence_indices = np.random.choice(len(vocabulary), size=sentence_length, replace=True)
    sentence = [vocabulary[idx] for idx in sentence_indices]
    
    # Generate input-output pairs for next word prediction
    data = []
    for i in range(len(sentence) - 1):
        input_word = sentence[:i+1]
        target_word = sentence[i+1]
        data.append((input_word, target_word))
    
    return data

# Example usage:
input_size = 10
hidden_size = 5
lstm_cell = LSTMCell(input_size, hidden_size)

# Generate training data
training_data = generate_data()

# Define vocabulary
vocabulary = ["after", "the", "initializing", "LSTM", "cell", "with", "a", "given", "input", "size", "and", "hidden", "we", "performed", "forward", "pass", "through", "updating", "hidden", "state", "based", "on", "at", "time", "step", "previous", "and", "cell", "performance", "by"]

# Training loop
for input_words, target_word in training_data:
    # Convert input words to one-hot vectors
    xt = np.zeros((1, input_size))
    for i, word in enumerate(input_words):
        xt[0, i] = 1  # one-hot encoding
        
    # Previous hidden state and cell state
    ht_1 = np.random.randn(1, hidden_size)
    Ct_1 = np.random.randn(1, hidden_size)
    
    # Forward pass through LSTM cell
    ht, Ct = lstm_cell.forward(xt, ht_1, Ct_1)
    
    # Perform prediction
    predicted_index = vocabulary.index(target_word)
    
    # Loss computation and backpropagation (not implemented here)
    # This part will depend on your specific training setup (e.g., using gradient descent)
    
    # Example: Print predicted index and target word
    print("Predicted Index:", predicted_index)
    print("Target Word:", target_word)
