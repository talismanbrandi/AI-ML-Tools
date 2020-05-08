import autograd.numpy as np
from autograd import grad, jacobian
import autograd.numpy.random as npr

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

nx = 20
ny = 20

dx = 1./nx
dy = 1./ny

x_space = np.linspace(0, 1, nx)
y_space = np.linspace(0, 1, ny)

#%%

def analtic_solution(x):
    return (1. / (np.exp(np.pi) - np.exp(-np.pi))) * \
            np.sin(np.pi * x[0]) * (np.exp(np.pi * x[1]) - np.exp(-np.pi * x[1]))

surface = np.zeros((nx,ny))

for i, x in enumerate(x_space):
    for j, y in enumerate(y_space):
        surface[i][j] = analtic_solution([x, y])

fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(x_space, y_space)
surf = ax.plot_surface(X, Y, surface, rstride=1, cstride=1, cmap=cm.rainbow, linewidth=0, antialiased=False)

ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,2)

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')

plt.show()

#%%

def f(x):
    return 0

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

def A(x):
    return x[1] * np.sin(np.pi * x[0])

def psy_trial(x, net_out):
    return A(x) + x[0] * (1. - x[0]) * x[1] * (1. - x[1]) * net_out

def loss_function(W, x, y):
        loss_sum = 0.

        for xi in x:
            for yi in y:
                input_point = np.array([xi, yi])
                net_out = neural_network(W, input_point)[0]

                psy_t_hessian = jacobian(jacobian(psy_trial))(input_point, net_out)

                gradient_of_trial_d2x = psy_t_hessian[0][0]
                gradient_of_trial_d2y = psy_t_hessian[1][1]

                func = f(input_point)

                err_sqr = ((gradient_of_trial_d2x + gradient_of_trial_d2y) - func)**2
                loss_sum += err_sqr
        return loss_sum - 0.01*np.dot(W[0].flatten(), W[0].flatten()) - 0.01*np.dot(W[1].flatten(), W[1].flatten())
#%%

W = [npr.randn(2, 20), npr.randn(20, 2), npr.randn(20), npr.randn(2)]
lmb = 0.001

for i in range(100):
    loss_grad = grad(loss_function)(W, x_space, y_space)

    W[0] = W[0] - lmb * loss_grad[0]
    W[1] = W[1] - lmb * loss_grad[1]
    W[2] = W[2] - lmb * loss_grad[2]
    W[3] = W[3] - lmb * loss_grad[3]

#%%

surface_sim = np.zeros((nx,ny))

for i, x in enumerate(x_space):
    for j, y in enumerate(y_space):
        net_out = neural_network(W, [x, y])[0]
        surface_sim[i][j] = psy_trial([x, y], net_out)

fig = plt.figure(figsize=(18,5))
ax = fig.add_subplot(1, 3, 1, projection='3d')
X, Y = np.meshgrid(x_space, y_space)
surf = ax.plot_surface(X, Y, surface, rstride=1, cstride=1, cmap=cm.rainbow,
        linewidth=0, antialiased=False)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1.5)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$\psi(x,y)$')
plt.title('analytic solution', pad=10)

ax = fig.add_subplot(1, 3, 2, projection='3d')
X, Y = np.meshgrid(x_space, y_space)
surf_sim = ax.plot_surface(X, Y, surface_sim, rstride=1, cstride=1, cmap=cm.rainbow,
        linewidth=0, antialiased=True)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1.5)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$\psi(x,y)$')
plt.title('ANN solution', pad=10)


ax = fig.add_subplot(1, 3, 3, projection='3d')
X, Y = np.meshgrid(x_space, y_space)
surf = ax.plot_surface(X, Y, surface_sim - surface, rstride=1, cstride=1, cmap=cm.rainbow,
        linewidth=0, antialiased=False)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(-0.075, 0.075)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$\Delta\psi(x, y)$')
plt.title('ANN - analytic', pad=10)
plt.suptitle(r"Solution to $\nabla^2\psi(x,y) = 0$", y=0.99)
plt.show()