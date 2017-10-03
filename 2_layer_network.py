
# coding: utf-8

# In[188]:


import numpy as np 
import pandas as pd


# In[189]:


def initialize_parameters(layers_dims):
    
    
    L = len(layers_dims)
    parameters = {}  #for storing W1,W2,b1,b2
    
    for i in range(1, L):
        parameters['W'+str(i)] = np.random.randn(layers_dims[i], layers_dims[i-1])*0.001
        parameters['b'+str(i)] = np.zeros((layers_dims[i], 1))
    
    return parameters
        
    
    


# In[190]:


def predict(x, parameters): 
    
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    Z1 = np.dot(W1, x) + b1
    A1 = relu(Z1)  
        
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    return A2


# In[191]:


def sigmoid(Z):
    A = np.divide(1, 1+np.exp(-Z))
    return A
    


# In[192]:


def relu(Z):
    A = Z*(Z>0)
    return A


# In[193]:


def compute_cost(A2, y):
    
    m = y.shape[1]
    cost = -np.sum((np.multiply(y, np.log(A2)) + np.multiply(1-y,np.log(1-A2))))/m
    
    return cost


# In[195]:


def two_layer_model(x, y, layers_dims, learning_rate=0.0075, itera=3000, print_cost=False):
    
    #parameters = initialize_parameters(layers_dims)
    n_input, n_hidden, n_output = layers_dims
    W1 = np.random.randn(n_hidden, n_input)
    b1 = np.zeros((n_hidden, 1))
    W2 = np.random.randn(n_output, n_hidden)
    b2 = np.zeros((n_output, 1))
    
    parameters = {}
    m = y.shape[1]
    costs = []
    
    for i in range(itera):
        
        A1 = sigmoid(np.dot(W1, x) + b1)  #relu 
        
        A2 = sigmoid(np.dot(W2, A1) + b2)  #sigmoid
        
        cost = compute_cost(A2, y)
        
        #initialize cost
        #dA2 = -np.divide(y, np.log(A2))+np.divide(1-y, np.log(1-A2))
        dZ2 = A2 - y
        dW2 = np.dot(dZ2, A1.T)/m
        db2 = np.sum(dZ2, axis=1, keepdims= True)/m
        
     
        dZ1 = np.dot(W2.T, dZ2) * (A1*(1-A1))
        dW1 = np.dot(dZ1, x.T)/m
        db1 = np.sum(dZ1, axis=1, keepdims= True)/m
        
        W1 = W1 - learning_rate*dW1 #np.dot(dZ1, x.T)/m
        b1 = b1 - learning_rate*db1 #np.sum(dZ1, axis=1, keepdims= True)/m
        W2 = W2 - learning_rate*dW2 #np.dot(dZ2, A1.T)/m
        b2 = b2 - learning_rate*db2 #np.sum(dZ2, axis=1, keepdims= True)/m
        
        
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i%100 == 0:
            costs.append(cost)
        
        
    parameters['W1'] = W1 
    parameters['b1'] = b1 
    parameters['W2'] = W2 
    parameters['b2'] = b2 
        
    plt.plot(costs)
    plt.show()
        
    return parameters, costs    
        
        


# In[196]:



train_x = np.random.randn(1, 20).reshape(1,20)
train_y = np.zeros((1, 20))
train_y[train_x>=0.1] = 1
train_y[train_x<0.1] = 0
train_y = train_y.reshape((1,20))
m = train_y.shape[1]
layers_dims = [1,3,1]


# In[197]:


import matplotlib.pyplot as plt

parameters, costs = two_layer_model(train_x, train_y, layers_dims, learning_rate=1, itera = 60000, print_cost=False)


# In[198]:


y_prob = predict(train_x, parameters)
y_pred = np.zeros((1, y_prob.shape[1]))
y_pred[y_prob>=0.5] = 1
y_pred[y_prob<0.5] = 0
y_pred = np.squeeze(y_pred)

from sklearn.metrics import accuracy_score
accuracy_score(np.squeeze(train_y), y_pred)


# In[ ]:




