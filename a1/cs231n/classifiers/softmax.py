from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax(x):
  if len(x.shape) == 1:
    x -= np.max(x) # for the numerical stability
    return np.exp(x)/sum(np.exp(x))
  elif len(x.shape) > 1:
    x -= np.max(x, axis=1, keepdims=True) # for the numerical stability
    return np.exp(x)/np.exp(x).sum(axis=1, keepdims=True)

def cross_entropy(p, Y):
  # delta = 1e-7
  return -np.sum(np.log(p) * Y)

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    num_classes = W.shape[1]
    for i in range(num_train):
      one_hot_y = np.zeros((num_classes,))
      one_hot_y[y[i]] = 1
      score = softmax(X[i] @ W)
      loss += cross_entropy(score, one_hot_y)
      dW[:,y[i]] -= X[i]
      for j in range(num_classes):
        dW[:,j] += X[i] * score[j]
    
    loss /= num_train
    loss += reg * np.sum(W * W)

    dW /= num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]

    scores = softmax(X @ W)
    one_hot_ys = np.zeros_like(scores)
    one_hot_ys[np.arange(num_train), y] = 1
    loss += cross_entropy(scores, one_hot_ys)

    scores += (-1) * one_hot_ys
    dW = X.T @ scores

    loss /= num_train
    loss += reg * np.sum(W * W)

    dW /= num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
