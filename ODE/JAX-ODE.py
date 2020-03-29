import jax.numpy as np
from jax import random
from jax import grad
from jax import vmap
from jax import jit
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def f(params, x):
    w0 = params[:10]
    b0 = params[10:20]
    w1 = params[20:30]
    b1 = params[30]
    x = sigmoid(x*w0 + b0)
    x = np.sum(x*w1) + b1
    return x

key = random.PRNGKey(0)
params = random.normal(key, shape=(31,))
dfdx = grad(f, 1)

f_vect = vmap(f, (None, 0))
dfdx_vect = vmap(dfdx, (None, 0))

def diff_eq(params, inputs):
    x = inputs
    return dfdx_vect(params, inputs) + 2.*inputs*f_vect(params, inputs)
    # return dfdx_vect(params, x) - (x**3 + 2.*x + x**2 * (1. + 3.*x**2) / (1. + x + x**3)) \
    #        + (x + (1. + 3.*x**2) / (1. + x + x**3)) * f_vect(params, x)
    # return dfdx_vect(params, x) - np.cos(x)

def initial_conditions(params, x, y):
    return f(params, x) - y

def f_analytic(x):
    return np.exp(-x ** 2)
    # return np.exp(-x**2/2.) / (1. + x + x**3) + x**2
    # return np.sin(x)


inputs = np.linspace(-2., 2., num=401)
epoch = 0
learning_rate = 0.01
momentum = 0.99
velocity = 0.

@jit
def loss(params, inputs):
    x = inputs
    eq = diff_eq(params, x)
    ic = initial_conditions(params, 0., 1.)
    return np.mean(eq**2) + ic**2

grad_loss = jit(grad(loss, 0))

loss_val = loss(params, inputs)
while loss_val > 1.e-4:
    epoch += 1
    loss_val = loss(params, inputs)
    if epoch % 100 == 0:
        print('epoch: %3d loss: %.6f' % (epoch, loss_val))
    gradient = grad_loss(params + momentum*velocity, inputs)
    velocity = momentum*velocity - learning_rate*gradient
    params += velocity

x = np.linspace(-2., 2., num=401)
plt.figure()
plt.plot(x, f_analytic(x), label='exact')
plt.plot(x, f_vect(params, x), label='ANN')
plt.legend()
plt.show()




