import autograd.numpy as np
import matplotlib.pyplot as plt
import autograd.numpy.random as ran

from autograd import grad
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

nx = 40
dx = 1./nx
#%%
# The differential equation
def f(x, psy, dpsy):
    '''
        d2(psy)/dx2 = f(x, dpsy/dx, psy)
        This is f() function on the right
    '''
    return -1./5. * np.exp(-x/5.) * np.cos(x) - 1./5. * dpsy - psy

def psy_analytic(x):
    '''
        Analytical solution of current problem
    '''
    return np.exp(-x/5.) * np.sin(x)

# The network
def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def sigmoid_grad(x):
    return sigmoid(x) * (1. - sigmoid(x))

def neural_network(W, x):
    a1 = sigmoid(np.dot(x, W[0]) + W[2])
    return np.dot(a1, W[1]) + W[3]

def neural_network_x(x):
    a1 = sigmoid(np.dot(x, W[0]) + W[2])
    return np.dot(a1, W[1]) + W[3]

def psy_trial(xi, net_out):
    return xi + xi**2 * net_out

psy_grad = grad(psy_trial)
psy_grad2 = grad(psy_grad)

def loss_function(W, x):
    loss_sum = 0.
    for xi in x:
        net_out = neural_network(W, xi)[0][0]

        psy_t = psy_trial(xi, net_out)
        gradient_of_trial = psy_grad(xi, net_out)
        second_gradient_of_trial = psy_grad2(xi, net_out)

        func = f(xi, psy_t, gradient_of_trial)
        err_sqr = (second_gradient_of_trial - func)**2
        loss_sum += err_sqr

    return loss_sum  - 0.1*np.dot(W[0].flatten(), W[0].flatten()) - 0.1*np.dot(W[1].flatten(), W[1].flatten())
#%%

x_space = np.linspace(0, 2, nx)
y_space = psy_analytic(x_space)

W = [ran.randn(1, 20), ran.randn(20, 1), ran.randn(20), ran.randn(1)]
lmb = 0.001

for i in range(1000):
    loss_grad = grad(loss_function)(W, x_space)

    W[0] = W[0] - lmb * loss_grad[0]
    W[1] = W[1] - lmb * loss_grad[1]
    W[2] = W[2] - lmb * loss_grad[2]
    W[3] = W[3] - lmb * loss_grad[3]
#%%
print(loss_function(W, x_space))
print(W)
res = [psy_trial(xi, neural_network(W, xi)[0][0]) for xi in x_space]

#%%

plt.figure(figsize=(8, 5))
plt.plot(x_space, y_space, c='g', label='analytic')
plt.plot(x_space, res, c='r', label='NN')
plt.legend()
plt.title(r'$\frac{d^2\psi(x)}{dx^2} + \frac{d\psi(x)}{dx} + \frac{1}{5}\psi(x) = \exp^{-\frac{x}{5}}\cos(x)$')
plt.xlabel(r'$x$')
plt.ylabel(r'$\psi(x)$')
plt.show()

