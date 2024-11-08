import numpy as np
import pickle
from math import sqrt
from numpy import exp
import time
import pickle
from scipy.optimize import minimize

def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon
    return W
    
def sigmoid(z):
    e = 1 / (1 + exp(-z))
    return e

def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    
    n = training_data.shape[0]
    # Feed forward code
    w1_transpose = np.transpose(w1)
    w2_transpose = np.transpose(w2)

    input_bias = np.ones(shape=(n, 1), dtype = np.float64) # create input bias 
    biased_training_data = np.append(training_data, input_bias, axis=1) # Add input bias to training data
    
    aj = np.dot(biased_training_data, w1_transpose) # Product of W and input data
    zj = sigmoid(aj) # Sigmoid of dot product
    
    hidden_bias = np.ones(shape=(zj.shape[0], 1), dtype = np.float64) 
    biased_zj = np.append(zj, hidden_bias, axis=1)
    
    bl = np.dot(biased_zj, w2_transpose)
    ol = sigmoid(bl)
    
    # Labelling output
    yl = np.zeros(shape=(n, 2), dtype = np.float64) # setting all output values to 0 initially
    
    for i in range(yl.shape[0]):   
        for j in range(yl.shape[1]):
            if j==training_label[i]:
                yl[i][j] = 1.0             #set the class labeled value to 1 and rest to 0    
        
    # Error function
    
    p = yl*np.log(ol) # Element wise multiplication
    q = (1-yl)*np.log(1-ol)
    sum1 = np.sum(p+q)
    constant = -1*n
    error = sum1/constant # -(yl*log(ol)+(1-y1)*log(1-ol))/n
    
    # Regularised error function  
    
    w1_square_sum = np.sum(np.square(w1))
    w2_square_sum = np.sum(np.square(w2))
    sum2 = w1_square_sum + w1_square_sum                
    reg_factor = (sum2*lambdaval)/(2*n)
    reg_error = error + reg_factor
    
    obj_val = reg_error # Regularised error w.r.t lambda
    
    
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.array([])
    delta_l = ol-yl
    delta_l_transpose = np.transpose(delta_l)
    
    grad_w2 = np.dot(delta_l_transpose, biased_zj)
    
    r = (1-biased_zj[:,0:n_hidden])*biased_zj[:,0:n_hidden]
    s = np.dot(delta_l, w2[:,0:n_hidden])
    rs = r*s
    rs_transpose = np.transpose(rs)
    grad_w1 = np.dot(rs_transpose, biased_training_data)
    
    # Regularised gradients
    
    reg_grad_w2 = (grad_w2 + (lambdaval*w2))/n
    reg_grad_w1 = (grad_w1 + lambdaval*w1)/n
    obj_grad = np.concatenate((reg_grad_w1.flatten(), reg_grad_w2.flatten()),0)
    
    return (obj_val, obj_grad)

def nnPredict(w1,w2,data):
    labels = np.zeros(shape=(data.shape[0], 1))
    #labels = np.zeros((data.shape[0],1))
    # Your code here

    # Feed forward code
    input_bias = np.ones(shape=(data.shape[0],1), dtype=np.float64)
    biased_data = np.append(data, input_bias, axis=1)
    
    w1_transpose = np.transpose(w1)
    w2_transpose = np.transpose(w2)
    
    aj = np.dot(biased_data, w1_transpose)
    zj = sigmoid(aj)
    
    hidden_bias = np.ones(shape=(zj.shape[0], 1), dtype=np.float64)
    biased_zj = np.append(zj, hidden_bias, axis=1)
    
    bl= np.dot(biased_zj, w2_transpose)
    ol = sigmoid(bl)
    
    for x in range(ol.shape[0]): # Label prediction
        max_arg = np.argmax(ol[x])
        labels[x] = max_arg
        
    return labels

def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
n_input = train_data.shape[1]

# set the number of nodes in output unit
n_class = 2

n_hidden_array = [4,8,12,16,20,24,28]

for x in n_hidden_array:
    for y in range(0,70,10):
        
        # set the number of nodes in hidden unit (not including bias unit)
        n_hidden = x # values - 4,8,12,16,20

        # set the regularization hyper-parameter
        lambdaval = y # values - 0 to 60, 10
        
        # Note current time
        t1 = time.time()
        
        # initialize the weights into some random matrices
        initial_w1 = initializeWeights(n_input, n_hidden);
        initial_w2 = initializeWeights(n_hidden, n_class);
        # unroll 2 weight matrices into single column vector
        initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
        
        args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

        #Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
        opts = {'maxiter' :50}    # Preferred value.

        nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
        params = nn_params.get('x')
        #Reshape nnParams from 1D vector into w1 and w2 matrices
        w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
        w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

        print("\n lambda:"+str(lambdaval))
        print("\n No of Hidden layers:"+str(n_hidden))
        #Test the computed parameters
        predicted_label = nnPredict(w1,w2,train_data)
        #find the accuracy on Training Dataset
        print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label.reshape(train_label.shape[0], 1)).astype(float))) + '%')
        predicted_label = nnPredict(w1,w2,validation_data)
        #find the accuracy on Validation Dataset
        print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label.reshape(validation_label.shape[0], 1)).astype(float))) + '%')
        predicted_label = nnPredict(w1,w2,test_data)
        #find the accuracy on Validation Dataset
        print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label.reshape(test_label.shape[0], 1)).astype(float))) + '%')
        
        t2 = time.time()
        
        print('\n Time taken:'+str(t2-t1))

