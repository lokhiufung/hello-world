
# coding: utf-8

# In[162]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 



def initialize_parameters(layers_dims):  
    L = len(layers_dims)  
    parameters = {}
    for i in range(L-1):
        parameters['W'+str(i+1)] = np.random.randn(layers_dims[i+1], layers_dims[i])
        parameters['b'+str(i+1)] = np.zeros((layers_dims[i+1], 1))       
            
    return parameters

def sigmoid(Z):
    
    A = 1/(1+np.exp(-Z))
    
    return A

def relu(Z):
    
    A = Z*(Z>0)
    
    return A

def forward_prop(x, parameters):
    caches = []
    A_prev = x 
    L = len(parameters)//2 +1   
    for i in range(1,L):
        Z = np.dot(parameters['W'+str(i)], A_prev) + parameters['b'+str(i)]
        A = sigmoid(Z)  #relu 
        caches.append((A_prev, Z))
        A_prev = A  
            
    return A, Z, caches
                

def compute_cost(AL, y):
    
    m = y.shape[1]
    cost = -np.sum(np.multiply(y, np.log(AL)) + np.multiply(1-y,np.log(1-AL)))/m
    
    return cost

def backward_prop(dAL, caches, parameters):
    
    m = dAL.shape[1]
    L = len(caches)+1
    dA = dAL
    grad = {}
    for i in reversed(range(1, L)):
        A, Z = caches[i-1]
        dZ = dA * (sigmoid(Z)*(1-sigmoid(Z)))
        grad['dW'+str(i)] = np.dot(dZ, A.T)/m
        grad['db'+str(i)] = np.sum(dZ, axis=1, keepdims= True)/m
        dA = np.dot(parameters['W'+str(i)].T, dZ)
                
    return grad

def gradient_checking(A, activation):
    
    numeric_grad = {}
    if activation == 'sigmoid':
        J = None
    return None

def update_parameters(grad, parameters, learning_rate):
    
    L = len(parameters)//2
    
    for i in range(L):
        parameters['W'+str(i+1)] = parameters['W'+str(i+1)] - learning_rate*grad['dW'+str(i+1)] 
        parameters['b'+str(i+1)] = parameters['b'+str(i+1)] - learning_rate*grad['db'+str(i+1)]
                
    return parameters

def predict(x, parameters):
    L = len(parameters)//2 +1   
    A_prev = x
    for i in range(1,L):
        Z = np.dot(parameters['W'+str(i)], A_prev) + parameters['b'+str(i)]
        A = sigmoid(Z)  #relu 
        A_prev = A  
        
    return A

def L_layer_model(x, y, layers_dims, learning_rate=0.0075, itera=3000, print_cost=False):
    
    parameters = initialize_parameters(layers_dims)
    m = y.shape[1]
    costs = []
   

    for i in range(itera):
        
        #forward_prop with all sigmoid
        AL, ZL, caches = forward_prop(x, parameters)
        
        #compute cost
        cost = compute_cost(AL, y)
        
        #initialize cost
        dAL = -np.divide(y, AL)+np.divide(1-y, 1-AL)
        
        #backward_prop
        grad = backward_prop(dAL, caches, parameters)
    
        #update parameters
        parameters = update_parameters(grad, parameters, learning_rate)
        
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        
        if i%100 == 0:
            costs.append(cost)
       
    plt.plot(costs)
    plt.show()
        
    return parameters    

