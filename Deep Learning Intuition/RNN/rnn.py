import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# generate some synthetic sequential data
np.random.seed(0)
seq_length = 20
num_samples = 1000
input_dim = 1


def generate_sequence(seq_length,num_samples,input_dim):
    
    # generate random sequence with Gaussian distribution
    x = np.random.randn(num_samples,seq_length,input_dim)
    
    # calculate the target column
    y = np.sum(x,axis=1)
    
    return x,y 


# generate training data
x_train, y_train = generate_sequence(seq_length,num_samples,input_dim)


# define the RNN model
model = tf.keras.Sequential([
    
    # simple RNN layer with 10 neurons
    tf.keras.layers.SimpleRNN(10,input_shape = (seq_length,input_dim)),
    
    # dense output layer with one neuron for regression
    tf.keras.layers.Dense(1)
])


# compile the model
model.compile(optimizer='adam',loss='mse')


# train the model
history = model.fit(x_train,y_train,epochs=50,batch_size=32,verbose=0)


# visualize the training loss
plt.plot(history.history['loss'])
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()