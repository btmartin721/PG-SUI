# -*- coding: utf-8 -*-
"""
@author: sohamghosh
"""


import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import numpy as np
from timeit import default_timer

def create_weights(sz):
    theta = theano.shared(np.array(np.random.rand(sz[0], sz[1]), dtype=theano.config.floatX))
    return theta
    
def grad_desc(cost, theta, alpha):
    return theta - (alpha * T.grad(cost, wrt=theta))

class Imputer:
    
    def __init__(self, X, reduced_dimensions, num_hidden_layers, hidden_layer_sizes=[]):
        assert(num_hidden_layers == len(hidden_layer_sizes) and num_hidden_layers > 0)
        self.X = X # set nan where not known
        self.X_reconstructed = None
        self.mask = np.isnan(self.X)
        self.complete_idx = np.where(~self.mask) # where data IS there
        self.l = num_hidden_layers
        self.V = np.random.randn(X.shape[0], reduced_dimensions)
        self.num_total_epochs = 0
        self.x_r = T.vector()
        self.learning_rate = T.scalar('eta')
        self.c = T.iscalar() # which index        
        self.r = T.iscalar() # which index
        
        self.V = theano.shared(np.array(np.random.rand(X.shape[0], reduced_dimensions), dtype=theano.config.floatX))
        self.weights = []
                
        self.U = theano.shared(np.array(np.random.rand(reduced_dimensions, X.shape[1]), dtype=theano.config.floatX))
        self.single_layer = nnet.sigmoid(T.dot(self.U.T, self.V[self.r, :]))
        
        
        self.layers = []
        # initialise
        for i in range(num_hidden_layers):
            if i == 0:
                self.weights.append(create_weights((reduced_dimensions, hidden_layer_sizes[0])))
            else:
                self.weights.append(create_weights((hidden_layer_sizes[i - 1], hidden_layer_sizes[i])))
        self.weights.append(create_weights((hidden_layer_sizes[-1], self.X.shape[1])))
                
        for i in range(num_hidden_layers):
            if i == 0:
                self.layers.append(nnet.sigmoid(T.dot(self.weights[i].T, self.V[self.r, :])))
            else:
                self.layers.append(nnet.sigmoid(T.dot(self.weights[i].T, self.layers[-1])))
        
        self.layers.append(nnet.sigmoid(T.dot(self.weights[-1].T, self.layers[-1])))
            
        self.fc1 = ((self.single_layer - self.x_r) ** 2)[self.c] # MSE error
        self.fc = ((self.layers[-1] - self.x_r) ** 2)[self.c] # MSE error
        
        self.phases = []
        
        self.phases.append(
            theano.function(inputs=[self.x_r, self.r, self.c, theano.In(self.learning_rate, value=0.1)], outputs= self.fc1, updates=[
                (self.U, grad_desc(self.fc1, self.U, self.learning_rate)), (self.V, grad_desc(self.fc1, self.V,  self.learning_rate))         
            ])
        )
        self.phases.append(
            theano.function(inputs=[self.x_r, self.r, self.c, theano.In(self.learning_rate, value=0.1)], outputs= self.fc, updates = [
                (theta, grad_desc(self.fc, theta, self.learning_rate)) for theta in self.weights ]
            )
        )
        
        self.phases.append(
            theano.function(inputs=[self.x_r, self.r, self.c, theano.In(self.learning_rate, value=0.1)], outputs= self.fc, updates = [
                (theta, grad_desc(self.fc, theta, self.learning_rate)) for theta in self.weights] + 
                    [(self.V, grad_desc(self.fc, self.V,  self.learning_rate))]
            )
        )
        
        self.run_phase1 = theano.function(inputs=[self.r], outputs=self.single_layer)
        self.run = theano.function(inputs=[self.r], outputs=self.layers[-1])
        
    def impute(self):        
        print "***********************************"
        print "* Running imputation"
        self.compute_reconstructed_X(phase=2)
        print '* Initial RMSE:', self.get_rmse()
        for i in range(1):
            print "* #### Phase ", (i + 1)
            self.initialise_parameters()
            while self.current_eta > self.target_eta:
                self.s = self.train_epoch(phase=i)
                if 1 - self.s/self.s_ < self.gamma:
                    self.current_eta = self.current_eta / 2
                    print '* reduced eta to', self.current_eta
                self.s_ = self.s
                self.num_epochs += 1
                self.num_total_epochs += 1
                self.display_num_epochs()
        
    def train_epoch(self, phase=1):
        start = default_timer()
        randomize = np.random.choice(len(self.complete_idx[0]), len(self.complete_idx[0]), replace=False)
        for r,c in zip(self.complete_idx[0][randomize], self.complete_idx[1][randomize]):
            self.phases[phase](self.X[r, :], r, c, self.current_eta)
        end = default_timer()
        print 'Epoch took', str((end-start)/60), 'm'
        self.compute_reconstructed_X()
        return self.get_rmse()     
        
    def display_num_epochs(self, interval=10):
        if self.num_epochs % interval == 0:
            print "* Epochs:", self.num_epochs, "\t", "RMSE:", self.s
    
    def compute_reconstructed_X(self, phase=2):
        self.X_reconstructed = np.zeros(self.X.shape)
        if phase == 2 or phase == 3:
            for r in range(self.X.shape[0]):
                self.X_reconstructed[r, :] = self.run(r)
        elif phase == 1:
            for r in range(self.X.shape[0]):
                self.X_reconstructed[r, :] = self.run_phase1(r)
        else:
            raise Exception('wrong phase')
            
    def initialise_parameters(self):
        self.initial_eta = 0.1
        self.target_eta =  0.0001
        self.s = 0
        self.s_ = np.inf
        self.current_eta = self.initial_eta
        self.gamma = 0.00001
        self.lambd = 0.0001
        self.num_epochs = 0
            
    def get_rmse(self):
        return np.sqrt(np.mean((self.X_reconstructed[~self.mask] - self.X[~self.mask]) ** 2))
