from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

print(tf.__version__)

nx_train = 1000
nx_test = 1000
learning_rate = 0.003
EPOCHS = 100000
patience = 300


class PrintDots(Callback):
    """ Print dots to monitor the progress of the fit
    """
    def on_epoch_end(self, epoch, logs):
        logs = logs or {}
        value = logs.get('val_loss')
        if epoch % 100 == 0:
            print(' epochs = ', epoch, ' val_loss = ', value)
        print('*', end='')


class TerminateOnBaseline(Callback):
    """ Callback that terminates training when monitored value reaches a specified baseline
    """
    def __init__(self, monitor='val_loss', patience=200):
        super(TerminateOnBaseline, self).__init__()
        self.monitor = monitor
        self.baseline = np.Inf
        self.patience = patience
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        value = logs.get(self.monitor)
        if epoch == 0:
            self.baseline = value/1000.
        if np.less(value, self.best):
            self.best = value
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
        if value is not None:
            if value <= self.baseline and self.wait >= self.patience:
                self.stopped_epoch = epoch
                print('\nepoch %d: Reached baseline, terminating training and lost patience' % epoch)
                self.model.stop_training = True
                print('Restoring model weights from the end of the best epoch.')
                self.model.set_weights(self.best_weights)
            elif self.wait >= self.patience:
                self.baseline *= 2.5
                self.wait = self.patience/2
                print('\nbaseline increased since learning times is approaching death')


def plot_history(history):
    """ Plots to track the history of the metrics
    """
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('mean abs error [y]')
    plt.plot(hist['epoch'], hist['mae'], label='train error')
    plt.plot(hist['epoch'], hist['val_mae'], label='val error')
    plt.ylim([0, 5])
    plt.legend()
    plt.show()

    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('mean square error [y$^2$]')
    plt.plot(hist['epoch'], hist['mse'], label='train error')
    plt.plot(hist['epoch'], hist['val_mse'], label='val error')
    plt.ylim([0, 1])
    plt.ylim([0, 20])
    plt.legend()
    plt.show()
    print(hist['loss'])


def printhistory(model, history, test_dataset, test_labels):
    """ Extract the history and make some plots
    """
    # plot_history(history)
    print("\n", "Epochs = ", history.epoch[-1])
    loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=0)
    print("Testing set mean square error: {:10.8f} y".format(mse))



def generatedata(nx1, nx2, label=1):
    """ Generate data for fitting
    """
    xa_train = np.zeros((nx1, 2))
    xa_test = np.zeros((nx1, 2))
    for i in range(0, nx1):
        xa_train[i, 0] = 1 + i/nx1 * 200
        if label == 1: xa_train[i, 1] = np.log(1./xa_train[i, 0]) * (1. + np.random.normal(0., 0.02, 1))
        if label == 2: xa_train[i, 1] = np.log(1. / xa_train[i, 0] + 0.1) * (1. + np.random.normal(0., 0.02, 1))
        if label == 3: xa_train[i, 1] = np.log(0.1) * (1. + np.random.normal(0., 0.02, 1))

    for i in range(0, nx2):
        xa_test[i, 0] = 1 + i/nx2 * 200
        if label == 1: xa_test[i, 1] = 1. / xa_test[i, 0]
        if label == 2: xa_test[i, 1] = 1. / xa_test[i, 0] + 0.1
        if label == 3: xa_test[i, 1] = 0.1

    return xa_train, xa_test

#%%


""" Create the dataframe and split it into training and testing sets
"""
train_data = pd.read_csv('/Users/ayanpaul/Codes/Interpolators/MatrixElelment/data/ggzz_grid.dat')
train_labels = train_data.pop(train_data.keys()[-1])

x_array = np.unique((train_data[train_data.keys()[0]]).to_numpy())
y_array = np.unique((train_data[train_data.keys()[1]]).to_numpy())
z_array = train_labels.to_numpy()

test_sizex = (len(x_array)-1)
test_sizey = (len(y_array)-1)
x_test = np.zeros(test_sizex*test_sizey, dtype='float32')
y_test = np.zeros(test_sizex*test_sizey, dtype='float32')
z_test = np.zeros(test_sizex*test_sizey, dtype='float32')

for i in range(0, test_sizex):
    for j in range(0, test_sizey):
        x_test[i * test_sizey + j] = (x_array[i] + x_array[i+1])/2.
        y_test[i * test_sizey + j] = (y_array[j] + y_array[j+1])/2.
        z_test[i * test_sizey + j] = (z_array[i * (test_sizey+1) + j] + z_array[(i + 1) * (test_sizey+1) + (j + 1)]) / 2.

test_data = pd.DataFrame({'pT': x_test, 'costheta': y_test, 'M': z_test})
test_labels = test_data.pop(test_data.keys()[-1])


#%%


def build_model(train_dataset):
    """ Build the model, define the optimizer and compile the model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(2, activation='linear', input_shape=[len(train_dataset.keys())]),
        tf.keras.layers.Dense(5, activation='sigmoid'),
        tf.keras.layers.Dense(5, activation='sigmoid'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


model = build_model(train_data)
model.summary()

#%%
""" Fit the model and extract the history of the fit
"""
history = model.fit(
    train_data, train_labels,
    epochs=EPOCHS, validation_split=0.2, verbose=0,
    callbacks=[TerminateOnBaseline(monitor='val_loss', patience=patience), PrintDots()])

printhistory(model, history, test_data, test_labels)

test_predictions = model.predict(test_data).flatten()

#%%

plt.figure()
ax = plt.gca()
plt.yscale('log')
plt.plot(train_x_S, np.exp(train_labels_S), color='#be0000', linestyle=':', label='S true')
plt.plot(test_values_S, np.exp(test_predictions_S), color='#be0000', linestyle='-', label='S trained')
plt.plot(train_x_B, np.exp(train_labels_B), color='#0048d8', linestyle=':', label='B true')
plt.plot(test_values_B, np.exp(test_predictions_B), color='#0048d8', linestyle='-', label='B trained')
plt.plot(train_x_BS, np.exp(train_labels_BS), color='#333333', linestyle=':', label='B - S true')
plt.plot(test_values_BS, test_predictions_BS, color='#333333', linestyle='-', label='B - S trained')
plt.xlabel('x')
plt.ylabel('P(x)')
plt.legend()
plt.savefig('fig1.pdf')
plt.show()

plt.figure()
# plt.yscale('log')
plt.plot(train_x_BS, np.exp(train_labels_BS), color='#333333', alpha=0.5, linestyle=':', label='B - S true')
plt.plot(test_values_BS, test_predictions_BS, color='#333333', linestyle='-', label='B - S trained')
plt.xlabel('x')
plt.ylabel('P(x)')
plt.ylim(0.08, 0.13)
plt.legend()
plt.savefig('fig2.pdf')
plt.show()


