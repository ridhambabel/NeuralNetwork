#NN with regularization, RelU, MSE loss
import numpy as np
import random

training_data = [(1,2), (6,45), (43, 95)]
rng = np.random.default_rng(seed=42)

class NeuralNetwork:

  def __init__(self, sizes):
    self.num_layers = len(sizes)
    self.sizes = sizes
    self.weights = [rng.normal(loc=0.0, scale=np.sqrt(2/x), size=(y,x)) for x,y in zip(sizes[:-1],sizes[1:])]
    self.biases = [rng.normal(loc=0.0, scale=1, size=(y,1)) for y in sizes[1:]]

  def feedforward(self,w,a,b):
    return (np.dot(w*a) + b)

  def SGD(self, training_data, epochs, mini_batch_size, eta, l2):
    n = len(training_data)
    for j in range(epochs):
      random.shuffle(training_data)
      mini_batches = [training_data[k*mini_batch_size:mini_batch_size]  for k in range(0,n,mini_batch_size)]
      for mini_batch in mini_batches:
        self.mini_batch_update(eta, mini_batch, l2)
      print(f"Epoch {j} complete")
      

  def mini_batch_update(self, eta, mini_batch, l2):
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    for x,y in mini_batch:
      delta_nabla_b, delta_nabla_w = self.backpropagation(x,y)
      nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
      nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
    self.biases = [b-eta*(nb/len(mini_batch)) for b,nb in zip(self.biases, nabla_b)]
    self.weights = [((1-l2)*w)-eta*(nw/len(mini_batch)) for w,nw in zip(self.weights, nabla_w)]


  def backpropagation(self, x, y):
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    # need to fill these with partial derivatives, mutliply by output loss by weights. This is bias, for weights multiply by activation
    # propagate forward then backward
    activation = x
    activations = [x]
    zs = []
    for w,b in zip(self.weights, self.biases):
      z = np.dot(w,activation) + b
      zs.append(z)
      activation = ReLU(z)
      activations.append(activation)

    #backward pass
    error = MSE(y,activations[-1])
    nabla_b[-1] = error
    nabla_w[-1] = np.dot(error, activations[-2].transpose)

    for l in range(2,self.num_layers):
      # error = np.dot(self.weights[-1].transpose(), error)
      # nabla_b[-l] = error
      # nabla_w[-l] = np.dot(error,activations[-l-1].transpose())

      nabla_b[-l] = np.dot(self.weights[-l+1].transpose(), nabla_b[-l+1])
      nabla_w[-l] = np.dot(nabla_b[-l],activations[-l-1].transpose())
    return nabla_b, nabla_w

def ReLU(z):
  return max(0,z)

def MSE(y,a):
  return (y-a)

# nn = NeuralNetwork([5,7,1])