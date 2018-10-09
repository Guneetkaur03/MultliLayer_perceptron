
# coding: utf-8

# In[17]:


import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


# In[25]:


import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import math

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def sigmoid_prime(x):
    f = 1/(1+np.exp(-x))
    return(f * (1 - f))

RANDOM_SEED = 1

np.random.seed(RANDOM_SEED)
epochs = 50000
input_size, hidden_size, output_size = 2, 3, 1
LR = 0.1 # learning rate

X = np.array([[1,0], [0,1],  [1,1],[0,0]])
y = np.array([ [1],   [1],    [0],[0]])

w_hidden = np.random.uniform(size=(input_size, hidden_size))
w_output = np.random.uniform(size=(hidden_size, output_size))


for epoch in range(epochs):
 
    # Forward
    act_hidden = sigmoid(np.dot(X, w_hidden))
    output = np.dot(act_hidden, w_output)
    
    # Calculate error
    error = y - output
    dZ = error * LR
    w_output += act_hidden.T.dot(dZ)
    dH = dZ.dot(w_output.T) * sigmoid_prime(act_hidden)
    w_hidden += X.T.dot(dH)
    


X_test =np.array([[1,1]])
act_hidden = sigmoid(np.dot(X_test, w_hidden))
print(np.dot(act_hidden, w_output))

X_test1 =np.array([[0,1]])
act_hidden = sigmoid(np.dot(X_test1, w_hidden))
print(np.dot(act_hidden, w_output))


# In[ ]:





# In[ ]:




