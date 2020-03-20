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


""" Create the dataframe and split it into training and testing sets
"""
xa_train_S, xa_test_S = generatedata(nx_train, nx_test, 1)
train_dataset_S = pd.DataFrame({'y': xa_train_S[:, 1], 'x': xa_train_S[:, 0]})
test_dataset_S = pd.DataFrame({'y': xa_test_S[:, 1], 'x': xa_test_S[:, 0]})
train_dataset_copy_S = train_dataset_S.copy()
test_dataset_copy_S = test_dataset_S.copy()
train_labels_S = train_dataset_S.pop('y')
test_labels_S = test_dataset_S.pop('y')

xa_train_B, xa_test_B = generatedata(nx_train, nx_test, 2)
train_dataset_B = pd.DataFrame({'y': xa_train_B[:, 1], 'x': xa_train_B[:, 0]})
test_dataset_B = pd.DataFrame({'y': xa_test_B[:, 1], 'x': xa_test_B[:, 0]})
train_dataset_copy_B = train_dataset_B.copy()
test_dataset_copy_B = test_dataset_B.copy()
train_labels_B = train_dataset_B.pop('y')
test_labels_B = test_dataset_B.pop('y')

xa_train_BS, xa_test_BS = generatedata(nx_train, nx_test, 3)
train_dataset_BS = pd.DataFrame({'y': xa_train_BS[:, 1], 'x': xa_train_BS[:, 0]})
test_dataset_BS = pd.DataFrame({'y': xa_test_BS[:, 1], 'x': xa_test_BS[:, 0]})
train_dataset_copy_BS = train_dataset_BS.copy()
test_dataset_copy_BS = test_dataset_BS.copy()
train_labels_BS = train_dataset_BS.pop('y')
test_labels_BS = test_dataset_BS.pop('y')


def build_model(train_dataset):
    """ Build the model, define the optimizer and compile the model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation='linear', input_shape=[len(train_dataset.keys())]),
        tf.keras.layers.Dense(5, activation='sigmoid'),
        tf.keras.layers.Dense(5, activation='sigmoid'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


model_S = build_model(test_dataset_S)
model_S.summary()

model_B = build_model(test_dataset_S)
model_B.summary()


""" Fit the model and extract the history of the fit
"""
history_S = model_S.fit(
    train_dataset_S, train_labels_S,
    epochs=EPOCHS, validation_split=0.2, verbose=0,
    callbacks=[TerminateOnBaseline(monitor='val_loss', patience=patience), PrintDots()])

history_B = model_B.fit(
    train_dataset_B, train_labels_B,
    epochs=EPOCHS, validation_split=0.2, verbose=0,
    callbacks=[TerminateOnBaseline(monitor='val_loss', patience=patience), PrintDots()])


def printhistory(model, history, test_dataset, test_labels):
    """ Extract the history and make some plots
    """
    # plot_history(history)
    print("\n", "Epochs = ", history.epoch[-1])
    loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=0)
    print("Testing set mean square error: {:10.8f} y".format(mse))

printhistory(model_S, history_S, test_dataset_S, test_labels_S)
printhistory(model_B, history_B, test_dataset_B, test_labels_B)

test_predictions_S = model_S.predict(test_dataset_S).flatten()
test_values_S = test_dataset_S.pop('x')
train_x_S = train_dataset_S.pop('x')

test_predictions_B = model_B.predict(test_dataset_B).flatten()
test_values_B = test_dataset_B.pop('x')
train_x_B = train_dataset_B.pop('x')

test_predictions_BS = np.exp(test_predictions_B) - np.exp(test_predictions_S)
test_values_BS = test_dataset_BS.pop('x')
train_x_BS = train_dataset_BS.pop('x')
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


