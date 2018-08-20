
# coding: utf-8

# # Programming assignment (Linear models, Optimization)
# 
# In this programming assignment you will implement a linear classifier and train it using stochastic gradient descent modifications and numpy.

# In[165]:

import numpy as np
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import math


# In[166]:

import sys
sys.path.append("..")
import grading
grader = grading.Grader(assignment_key="UaHtvpEFEee0XQ6wjK-hZg", 
                      all_parts=["xU7U4", "HyTF6", "uNidL", "ToK7N", "GBdgZ", "dLdHG"])


# In[167]:

# token expires every 30 min
COURSERA_EMAIL = "-------------"
COURSERA_TOKEN = "---------"


# ## Two-dimensional classification
# 
# To make things more intuitive, let's solve a 2D classification problem with synthetic data.

# In[168]:

with open('train.npy', 'rb') as fin:
    X = np.load(fin)
    
with open('target.npy', 'rb') as fin:
    y = np.load(fin)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, s=20)
plt.show()


# # Task
# 
# ## Features
# 
# As you can notice the data above isn't linearly separable. Since that we should add features (or use non-linear model). Note that decision line between two classes have form of circle, since that we can add quadratic features to make the problem linearly separable. The idea under this displayed on image below:
# 
# ![](kernel.png)

# In[169]:

def expand(X):
    """
    Adds quadratic features. 
    This expansion allows your linear model to make non-linear separation.
    
    For each sample (row in matrix), compute an expanded row:
    [feature0, feature1, feature0^2, feature1^2, feature0*feature1, 1]
    
    :param X: matrix of features, shape [n_samples,2]
    :returns: expanded features of shape [n_samples,6]
    """
    X_expanded = np.zeros((X.shape[0], 6))
    X_expanded[:,0], X_expanded[:,1] = X[:,0],X[:,1]
    X_expanded[:,2], X_expanded[:,3]= X[:,0]**2, X[:,1]**2
    X_expanded[:,4], X_expanded[:,5] = X[:,0]*X[:,1], np.ones(X.shape[0])
    return  X_expanded


# In[170]:

X_expanded = expand(X)


# Here are some tests for your implementation of `expand` function.

# In[171]:

# simple test on random numbers

dummy_X = np.array([
        [0,0],
        [1,0],
        [2.61,-1.28],
        [-0.59,2.1]
    ])

# call your expand function
dummy_expanded = expand(dummy_X)

# what it should have returned:   x0       x1       x0^2     x1^2     x0*x1    1
dummy_expanded_ans = np.array([[ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  1.    ],
                               [ 1.    ,  0.    ,  1.    ,  0.    ,  0.    ,  1.    ],
                               [ 2.61  , -1.28  ,  6.8121,  1.6384, -3.3408,  1.    ],
                               [-0.59  ,  2.1   ,  0.3481,  4.41  , -1.239 ,  1.    ]])

#tests
assert isinstance(dummy_expanded,np.ndarray), "please make sure you return numpy array"
assert dummy_expanded.shape == dummy_expanded_ans.shape, "please make sure your shape is correct"
assert np.allclose(dummy_expanded,dummy_expanded_ans,1e-3), "Something's out of order with features"

print("Seems legit!")


# ## Logistic regression
# 
# To classify objects we will obtain probability of object belongs to class '1'. To predict probability we will use output of linear model and logistic function:
# 
# $$ a(x; w) = \langle w, x \rangle $$
# $$ P( y=1 \; \big| \; x, \, w) = \dfrac{1}{1 + \exp(- \langle w, x \rangle)} = \sigma(\langle w, x \rangle)$$
# 

# In[172]:

def probability(X, w):
    """
    Given input features and weights
    return predicted probabilities of y==1 given x, P(y=1|x), see description above
        
    Don't forget to use expand(X) function (where necessary) in this and subsequent functions.
    
    :param X: feature matrix X of shape [n_samples,6] (expanded)
    :param w: weight vector w of shape [6] for each of the expanded features
    :returns: an array of predicted probabilities in [0,1] interval.
    """
    z = np.dot(X,w)
    
    a = 1./(1+np.exp(-z))
    
    return np.array(a)


# In[173]:

dummy_weights = np.linspace(-1, 1, 6)
ans_part1 = probability(X_expanded[:1, :], dummy_weights)[0]


# In[174]:

## GRADED PART, DO NOT CHANGE!
grader.set_answer("xU7U4", ans_part1)


# In[175]:

# you can make submission with answers so far to check yourself at this stage
grader.submit(COURSERA_EMAIL, COURSERA_TOKEN)


# In logistic regression the optimal parameters $w$ are found by cross-entropy minimization:
# 
# $$ L(w) =  - {1 \over \ell} \sum_{i=1}^\ell \left[ {y_i \cdot log P(y_i = 1 \, | \, x_i,w) + (1-y_i) \cdot log (1-P(y_i = 1\, | \, x_i,w))}\right] $$
# 
# 

# In[182]:

def compute_loss(X, y, w):
    """
    Given feature matrix X [n_samples,6], target vector [n_samples] of 1/0,
    and weight vector w [6], compute scalar loss function using formula above.
    """
    l = X.shape[0]
    
    a = probability(X, w)
    
    cross_entropy = y*np.log(a) +(1-y)*np.log(1-a)
    cost = -np.sum(cross_entropy)/float(l)
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost
    # TODO:<your code here>


# In[183]:

# use output of this cell to fill answer field 
ans_part2 = compute_loss(X_expanded, y, dummy_weights)


# In[184]:

## GRADED PART, DO NOT CHANGE!
grader.set_answer("HyTF6", ans_part2)


# In[185]:

# you can make submission with answers so far to check yourself at this stage
grader.submit(COURSERA_EMAIL, COURSERA_TOKEN)


# Since we train our model with gradient descent, we should compute gradients.
# 
# To be specific, we need a derivative of loss function over each weight [6 of them].
# 
# $$ \nabla_w L = ...$$
# 
# We won't be giving you the exact formula this time — instead, try figuring out a derivative with pen and paper. 
# 
# As usual, we've made a small test for you, but if you need more, feel free to check your math against finite differences (estimate how $L$ changes if you shift $w$ by $10^{-5}$ or so).

# In[186]:

def compute_grad(X, y, w):
    """
    Given feature matrix X [n_samples,6], target vector [n_samples] of 1/0,
    and weight vector w [6], compute vector [6] of derivatives of L over each weights.
    """
    m = X.shape[0]
    A = probability(X, w)
    dZ = A - y
    #cost = compute_loss(X, y, w)
    dW = np.dot(dZ, X) / float(m)
    return dW
    
    # TODO<your code here>


# In[187]:

# use output of this cell to fill answer field 
ans_part3 = np.linalg.norm(compute_grad(X_expanded, y, dummy_weights))


# In[188]:

## GRADED PART, DO NOT CHANGE!
grader.set_answer("uNidL", ans_part3)


# In[189]:

# you can make submission with answers so far to check yourself at this stage
grader.submit(COURSERA_EMAIL, COURSERA_TOKEN)


# Here's an auxiliary function that visualizes the predictions:

# In[190]:

from IPython import display

h = 0.01
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

def visualize(X, y, w, history):
    """draws classifier prediction with matplotlib magic"""
    Z = probability(expand(np.c_[xx.ravel(), yy.ravel()]), w)
    Z = Z.reshape(xx.shape)
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    
    plt.subplot(1, 2, 2)
    plt.plot(history)
    plt.grid()
    ymin, ymax = plt.ylim()
    plt.ylim(0, ymax)
    display.clear_output(wait=True)
    plt.show()


# In[191]:

visualize(X, y, dummy_weights, [0.5, 0.5, 0.25])


# ## Training
# In this section we'll use the functions you wrote to train our classifier using stochastic gradient descent.
# 
# You can try change hyperparameters like batch size, learning rate and so on to find the best one, but use our hyperparameters when fill answers.

# ## Mini-batch SGD
# 
# Stochastic gradient descent just takes a random example on each iteration, calculates a gradient of the loss on it and makes a step:
# $$ w_t = w_{t-1} - \eta \dfrac{1}{m} \sum_{j=1}^m \nabla_w L(w_t, x_{i_j}, y_{i_j}) $$
# 
# 

# In[192]:

# please use np.random.seed(42), eta=0.1, n_iter=100 and batch_size=4 for deterministic results

np.random.seed(42)
w = np.array([0, 0, 0, 0, 0, 1])

eta= 0.1 # learning rate

n_iter = 100
batch_size = 4
loss = np.zeros(n_iter)
plt.figure(figsize=(12, 5))

for i in range(n_iter):
    ind = np.random.choice(X_expanded.shape[0], batch_size)
    loss[i] = compute_loss(X_expanded, y, w)
    if i % 10 == 0:
        visualize(X_expanded[ind, :], y[ind], w, loss)
    grad = compute_grad(X_expanded[ind, :], y[ind], w)
    w = w - eta * grad
    # TODO:<your code here>

visualize(X, y, w, loss)
plt.clf()


# In[193]:

# use output of this cell to fill answer field 

ans_part4 = compute_loss(X_expanded, y, w)


# In[194]:

## GRADED PART, DO NOT CHANGE!
grader.set_answer("ToK7N", ans_part4)


# In[195]:

# you can make submission with answers so far to check yourself at this stage
grader.submit(COURSERA_EMAIL, COURSERA_TOKEN)


# ## SGD with momentum
# 
# Momentum is a method that helps accelerate SGD in the relevant direction and dampens oscillations as can be seen in image below. It does this by adding a fraction $\alpha$ of the update vector of the past time step to the current update vector.
# <br>
# <br>
# 
# $$ \nu_t = \alpha \nu_{t-1} + \eta\dfrac{1}{m} \sum_{j=1}^m \nabla_w L(w_t, x_{i_j}, y_{i_j}) $$
# $$ w_t = w_{t-1} - \nu_t$$
# 
# <br>
# 
# 
# ![](sgd.png)
# 

# In[196]:

# please use np.random.seed(42), eta=0.05, alpha=0.9, n_iter=100 and batch_size=4 for deterministic results
np.random.seed(42)
w = np.array([0, 0, 0, 0, 0, 1])

eta = 0.05 # learning rate
alpha = 0.9 # momentum
nu = np.zeros_like(w)

n_iter = 100
batch_size = 4
loss = np.zeros(n_iter)
plt.figure(figsize=(12, 5))

for i in range(n_iter):
    ind = np.random.choice(X_expanded.shape[0], batch_size)
    loss[i] = compute_loss(X_expanded, y, w)
    if i % 10 == 0:
        visualize(X_expanded[ind, :], y[ind], w, loss)
    grad = compute_grad(X_expanded[ind, :], y[ind], w)
    nu = [ (alpha*nval)+(eta*grad[ix]) for ix,nval in enumerate(nu)]
    w = [ wt - nu[ix] for ix,wt in enumerate(w)]

    # TODO:<your code here>

visualize(X, y, w, loss)
plt.clf()


# In[197]:

# use output of this cell to fill answer field 
ans_part5 = compute_loss(X_expanded, y, w)


# In[198]:

## GRADED PART, DO NOT CHANGE!
grader.set_answer("GBdgZ", ans_part5)


# In[199]:

# you can make submission with answers so far to check yourself at this stage
grader.submit(COURSERA_EMAIL, COURSERA_TOKEN)


# ## RMSprop
# 
# Implement RMSPROP algorithm, which use squared gradients to adjust learning rate:
# 
# $$ G_j^t = \alpha G_j^{t-1} + (1 - \alpha) g_{tj}^2 $$
# $$ w_j^t = w_j^{t-1} - \dfrac{\eta}{\sqrt{G_j^t + \varepsilon}} g_{tj} $$

# In[205]:

np.random.seed(42)
w = np.array([0, 0, 0, 0, 0, 1.])
eta = 0.1 # learning rate
alpha = 0.9 # moving average of gradient norm squared
g2 = np.zeros_like(w)
eps = 1e-8

n_iter = 100
batch_size = 4
loss = np.zeros(n_iter)
plt.figure(figsize=(12,5))
G = np.zeros_like(w)

for i in range(n_iter):
    ind = np.random.choice(X_expanded.shape[0], batch_size)
    loss[i] = compute_loss(X_expanded, y, w)
    if i % 10 == 0:
        visualize(X_expanded[ind, :], y[ind], w, loss)
    grad = compute_grad(X_expanded[ind, :], y[ind], w)
    g2 = grad**2
    G = alpha*G+(1-alpha)*g2
    
    w = w - eta*grad/np.sqrt(G+eps)
    # TODO:<your code here>

visualize(X, y, w, loss)
plt.clf()



# In[206]:

ans_part6 = compute_loss(X_expanded, y, w)


# In[207]:

grader.set_answer("dLdHG", ans_part6)


# In[208]:

grader.submit(COURSERA_EMAIL, COURSERA_TOKEN)

