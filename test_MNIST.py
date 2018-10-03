#!/usr/bin/python

from Data import *
import numpy as np
from NeuralNetwork import *
#from hw2q2 import *

def main():
  mnist=Data.fromName('MNIST')
  mnist.normalize()
  mnist.toOneHot()

  np.random.seed(1)

  M=1E-1  # initial weight size
  nInputs=784
  layers=[(50, ReLU), (50, ReLU), (10, Linear)]
  CE=ObjectiveFunction('crossEntropyLogit')
  nn=NeuralNetwork(nInputs, layers, M)

  nIter=10000
  B=100
  eta=0.1 # learning rate
  for i in range(nIter):
    x, y = mnist.next_batch(B)
    logit = nn.doForward(x)
    J=CE.doForward(logit, y)
    dp=CE.doBackward(y)
    nn.doBackward(dp)
    nn.updateWeights(eta)

    # compute error rate
    if (i%100==0):
      p=nn.doForward(mnist.x_test)
      yhat=p.argmax(axis=0)
      yTrue=mnist.y_test.argmax(axis=0)
      accu = float(sum(yhat==yTrue))/len(yTrue)
      print( '\riter %d, J=%f, accu=%.2f' % (i, J, accu))

main()
# run: python %
