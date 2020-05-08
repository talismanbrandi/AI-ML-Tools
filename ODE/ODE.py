import autograd.numpy as np
import matplotlib.pyplot as plt
import autograd.numpy.random as ran

from autograd import  grad

nx = 60
dx = 1./nx

# The differential equation
def A(x):
    '''
    Left part of the initial equation
    '''
    return x + (1. + 3.*x**2) / (1. + x + x**3)

def B(x):
    '''
    Right part of the initial equation
    '''
    return x**3 + 2.*x + x**2 * (1. + 3.*x**2) / (1. + x + x**3)

def f(x, psy):
    '''
        d(psy)/dx = f(x, psy)
        This is f() function on the right
    '''
    return B(x) - psy * A(x)

def psy_analytic(x):
    '''
        Analytical solution of current problem
    '''
    return np.exp(-x**2/2.) / (1. + x + x**3) + x**2

# The network
def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def sigmoid_grad(x):
    return sigmoid(x) * (1. - sigmoid(x))

def neural_network(W, x):
    a1 = sigmoid(np.dot(x, W[0]))
    return np.dot(a1, W[1])

def d_neural_network_dx(W, x, k=1):
    return np.dot(np.dot(W[1].T, W[0].T**k), sigmoid_grad(x))

def loss_function(W, x):
    loss_sum = 0.
    for xi in x:
        net_out = neural_network(W, xi)[0][0]
        psy_t = 1. + xi * net_out
        d_net_out = d_neural_network_dx(W, xi)[0][0]
        d_psy_t = net_out + xi * d_net_out
        func = f(xi, psy_t)
        err_sqr = (d_psy_t - func)**2

        loss_sum += err_sqr
    return loss_sum

#%%

x_space = np.linspace(0, 1, nx)
y_space = psy_analytic(x_space)
psy_fd = np.zeros_like(x_space)
psy_fd[0] = 1.

for i in range(1, len(x_space)):
    psy_fd[i] = psy_fd[i-1] + B(x_space[i]) * dx - psy_fd[i-1] * A(x_space[i]) * dx

plt.figure()
plt.plot(x_space, y_space)
plt.plot(x_space, psy_fd)
plt.show()

#%%

W = [ran.randn(1, 7), ran.randn(7, 1)]
lmb = 0.001

for i in range(1000):
    loss_grad = grad(loss_function)(W, x_space)

    W[0] = W[0] - lmb * loss_grad[0]
    W[1] = W[1] - lmb * loss_grad[1]

print(loss_function(W, x_space))

res = [1. + xi * neural_network(W, xi)[0][0] for xi in x_space]

#%%

plt.figure()
plt.plot(x_space, y_space, c='g', label='analytic')
plt.plot(x_space, psy_fd, c='b', label='numeric')
plt.plot(x_space, res, c='r', label='NN')
plt.legend()
plt.show()