from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math as m

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

print(tf.__version__)

nx = 1001
learning_rate = 0.003
EPOCHS = 100000
patience = 300

xa = np.zeros((nx, 2))

""" Generate data for fitting
"""
for i in range (0, nx):
    xa[i, 0] = i * 2. * m.pi/(nx - 1.) + 0.01
    # xa[i, 1] = 2. + 3.* xa[i, 0]
    xa[i, 1] = m.sin(xa[i, 0]) * m.cos(4.*xa[i, 0]) * (1. + np.random.normal(0., 0.2, 1))
    # xa[i, 1] = m.sin(2.*xa[i, 0])/2./xa[i, 0]
    # xa[i, 1] = m.exp(xa[i, 0])*m.sin(xa[i, 0])


""" Create the dataframe and split it into training and testing sets
"""
dataset = pd.DataFrame({'y': xa[:, 1], 'x': xa[:, 0]})
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
train_dataset_copy = train_dataset.copy()
train_labels = train_dataset.pop('y')
test_labels = test_dataset.pop('y')


def build_model():
    """ Build the model, define the optimizer and compile the model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation='linear', input_shape=[len(train_dataset.keys())]),
        tf.keras.layers.Dense(16, activation='tanh'),
        tf.keras.layers.Dense(16, activation='selu'),
        tf.keras.layers.Dense(16, activation='tanh'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


model = build_model()
model.summary()


class PrintDots(Callback):
    """ Print dots to monitor the progress of the fit
    """
    def on_epoch_end(self, epoch, logs):
        logs = logs or {}
        value = logs.get('val_loss')
        if epoch % 100 == 0:
            print(' epochs = ', epoch, ' val_loss = ', value)
        print('*', end='')


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


""" Extract the history of the fit
"""
history = model.fit(
    train_dataset, train_labels,
    epochs=EPOCHS, validation_split = 0.2, verbose=0,
    callbacks=[TerminateOnBaseline(monitor='val_loss', patience=patience), PrintDots()])


""" Extract the history and make some plots
"""
# plot_history(history)
print("\n", "Epochs = ", history.epoch[-1])

loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=0)
print("Testing set mean square error: {:10.8f} y".format(mse))

test_predictions = model.predict(test_dataset).flatten()

test_values = test_dataset.pop('x')
plt.figure()
ax = plt.gca()
train_dataset_copy.plot(kind='scatter', x='x', y='y', ax=ax)
plt.scatter(test_values, test_predictions, color='red')
plt.show()
