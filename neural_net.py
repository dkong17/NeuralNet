import numpy as np
from matplotlib import pyplot as plt
import pdb

class NeuralNet:

    def __init__(self):
        # Random weights
        self.V = 0.1 * np.random.randn(200, 785) #(200, 784 + 1)
        self.W = 0.1 * np.random.randn(26, 201) #(26, 200 + 1)

    /** Shortcut to use saved numpy objects. */
    def load_weights(self, filename):
        self.V, self.W = np.load(filename)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def tanh_prime(self, x):
        return 1.0 - np.tanh(x)**2

    def forward(self, X, V=None, W=None):
        # Propogate inputs though network
        if V is None and W is None:
            V = self.V
            W = self.W
        self.Vx = np.dot(V, X)
        self.h = np.tanh(self.Vx)
        self.h = np.append(self.h, 1) # add bias to hidden layer (201, 1)
        Wh = np.dot(self.W, self.h) # (26, 1)
        self.z = self.sigmoid(Wh)
        return self.z

    def cross_entropy_loss(self, X, y):
        self.z = self.forward(X)    
        return -sum(np.multiply(y.ravel() ,np.log(self.z)) + np.multiply(1.0 - y.ravel(), np.log(1.0-self.z)))

    def cross_entropy_loss_prime(self, X, y):
        self.forward(X)
        t1 = np.reshape(self.z - np.ravel(y), (26,1)) #(26, 1)
        dJdW = np.dot(t1, self.h[np.newaxis])
        dLdh = np.dot(self.W.T, t1)
        dLdh = np.reshape(np.delete(dLdh, 200), (200, 1))
        dhdV = self.tanh_prime(self.Vx)
        #Do element wise mutliplication of dhdV with dLdh (200, 1)
        dJdV = np.multiply(dLdh, dhdV)
        #Dot with input (technically part of dhdV) (200, 785)
        dJdV = np.dot(dJdV, X.T)
        return dJdV, dJdW

    def train(self, images, labels, **params):
        max_iterations = float('inf') if 'max_iterations' not in params else params['max_iterations']
        max_epochs = float("inf") if 'max_epochs' not in params else params['max_epochs']
        alphaV = 0.01 if 'alphaV' not in params else params['alphaV']
        alphaW = 0.01 if 'alphaW' not in params else params['alphaW']
        decayV = 1 if 'decayV' not in params else params['decayV']
        decayW = 1 if 'decayW' not in params else params['decayW']
        decay_epoch = 0 if 'decay_epoch' not in params else params['decay_epoch']
        if 'visualize' in params:
            visualize = True
            plot_iter = params['plot_iter']
            losses = []
        epoch_count = 0
        iter_count = 0
        true_count = 0
        while (iter_count < max_iterations and epoch_count < max_epochs):
            X = np.reshape(images[iter_count], (785, 1))
            y = np.reshape(labels[iter_count], (26, 1))
            dJdV, dJdW = self.cross_entropy_loss_prime(X, y)
            self.V -= alphaV * dJdV
            self.W -= alphaW * dJdW
            if visualize and true_count % plot_iter == 0:
                losses.append(self.cross_entropy_loss(X, y))
            iter_count += 1
            true_count += 1
            if iter_count == len(images):
                iter_count = 0
                epoch_count += 1
                if decay_epoch > 0 and epoch_count % decay_epoch == 0:
                    alphaV *= decayV
                    alphaW *= decayW
        np.save("weights.npy", np.array((self.V, self.W)))
        if visualize:
            plt.figure()
            x_axis = [plot_iter * i for i in range(len(losses))]
            plt.plot(x_axis, losses)
            plt.xlabel('Number of Iterations')
            plt.ylabel('Cross Entropy Loss')
            plt.title('Loss vs Iterations')
            plt.grid(True)
            plt.savefig('losses.png')

    def predict(self, images, labels=None):
        results = []
        error = 0
        for image in images:
            image = np.reshape(image, (785, 1))
            z = self.forward(image)
            results.append(np.argmax(z))
        if labels is not None:
            for i in range(len(labels)):
                if labels[i][results[i]] == 0:
                    error += 1
            return results, error/len(labels)
        return results