import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

def nanmean_squared_error(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    residuals = y_true - y_pred
    residuals_no_nan = tf.where(
        tf.math.is_nan(residuals), 
        tf.zeros_like(residuals), 
        residuals
    )
    sum_residuals = tf.reduce_sum(
        tf.square(residuals_no_nan), -1
    ) / tf.reduce_sum(
        tf.cast(~tf.math.is_nan(y_true), tf.float64), -1
    )
    return sum_residuals

def negative_correlation_loss(y_true, y_pred, lambda_nc=0.1):
    y_pred_mean = tf.reduce_mean(y_pred, axis=0)
    mse_loss = nanmean_squared_error(y_true, y_pred)
    correlation_loss = tf.reduce_mean(tf.square(y_pred - y_pred_mean))
    return mse_loss + lambda_nc * correlation_loss

class NegativeCorrelationLearning:

    def __init__(self, n_models, input_shape, lambda_nc=0.1):
        self.n_models = n_models
        self.models = [self.build_model(input_shape) for _ in range(n_models)]
        self.lambda_nc = lambda_nc
        self.optimizer = Adam()

    def build_model(self, input_shape):
        model = Sequential()
        model.add(Dense(units=128, activation='relu', input_shape=(input_shape,)))
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=16, activation='relu'))
        model.add(Dense(units=8, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(units=1))
        return model

    def train(self, X_train, y_train, epochs=100, batch_size=32):
        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

        for epoch in range(epochs):
            for step, (x_batch_train, y_batch_train) in enumerate(dataset):
                with tf.GradientTape(persistent=True) as tape:
                    y_preds = [model(x_batch_train, training=True) for model in self.models]
                    y_preds_stack = tf.stack(y_preds, axis=0)
                    loss = sum(
                        negative_correlation_loss(y_batch_train, y_pred, self.lambda_nc) 
                        for y_pred in y_preds_stack
                    )

                grads = [
                    tape.gradient(loss, model.trainable_weights) 
                    for model in self.models
                ]
                for model, grad in zip(self.models, grads):
                    self.optimizer.apply_gradients(zip(grad, model.trainable_weights))

    def predict(self, X):
        y_preds = [model.predict(X) for model in self.models]
        return np.mean(y_preds, axis=0)

def nanmean_squared_error(y_true, y_pred):
    y_pred = ops.convert_to_tensor_v2(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    
    residuals = y_true - y_pred
    
    residuals_no_nan = tf.where(
        tf.math.is_nan(residuals), 
        tf.zeros_like(residuals), 
        residuals
    )
    
    sum_residuals = tf.reduce_sum(
        math_ops.squared_difference(residuals_no_nan, 0), axis=-1
    ) / tf.reduce_sum(
        tf.cast(~tf.math.is_nan(y_true), tf.float64), axis=-1
    )
    
    return sum_residuals

def evaluate_model(trainX, trainy, testX, testy):

    model = Sequential()
    model.add(Dense(units=128, activation='relu')) 
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=6))

    model.compile(loss=nanmean_squared_error, optimizer='adam')
    history = model.fit(
        trainX, trainy, 
        epochs=200, 
        batch_size=64, 
        validation_data=(testX, testy), 
        verbose=0
    )

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = len(loss)

    predy = model.predict(testX)
    test_r2 = np.zeros(6)
    for i in range(6):
        flag1 = ~np.isnan(testy[:, i])
        test_r2[i] = r2_score(
            testy[:, i][flag1], 
            predy[:, i][flag1], 
            multioutput='raw_values'
        )
    
    return model, test_r2

def ensemble_predictions(members, testX):
    
    yhats = [model.predict(testX) for model in members]
    yhats = np.array(yhats)
    
    predictions = np.mean(yhats, axis=0)
    variances = np.var(yhats, axis=0)
    
    return predictions, variances

def evaluate_n_members(members, n_members, testX, testy):
    
    subset = members[:n_members]
    
    yhat, variances = ensemble_predictions(subset, testX)
    
    avg_var = np.mean(variances, axis=0)
    
    # Calculate R2 scores for each output dimension
    test_r2 = np.zeros(6)
    for i in range(6):
        flag1 = ~np.isnan(testy[:, i])
        test_r2[i] = r2_score(testy[:, i][flag1], yhat[:, i][flag1], multioutput='raw_values')
    
    return test_r2, avg_var

def main():
    # Read in the training data
    dataset_a_smiles_p = pd.read_csv(r"\Smiles.csv")
    dataset_a_grouped = dataset_a_smiles_p.groupby('Smiles').mean().reset_index()

    Y = dataset_a_grouped.iloc[:, -6:-1]

    # Normalize Y
    Y = np.array(Y)
    scaler = StandardScaler()
    Y = scaler.fit_transform(Y)

    X = pd.read_csv(r"\Dataset\Molecular_Fingerprints.csv")

    X = np.array(X)
    x_scaler = StandardScaler()
    X = x_scaler.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

    newX = X_test
    newy = Y_test

    n_splits = 16
    scores, members = list(), list()

    ncl = NegativeCorrelationLearning(n_models=5, input_shape=X_train.shape[1])
    ncl.train(X_train, Y_train, epochs=100, batch_size=32)
    y_pred = ncl.predict(X_test)
    y_pred = y_pred.reshape(-1, 1)

    # Summarize expected performance
    print('-----------------------------------------')
    print('Expected R2: %.3f' % np.mean(scores))
    print('Mean R2: ' + str(np.mean(scores, axis=0)))
    print('Estimated Std of R2: ' + str(np.std(scores, axis=0)))
    print('-----------------------------------------')

    single_scores, ensemble_scores = list(), list()
    for i in range(1, n_splits+1):
        ensemble_score, ensemble_variance = evaluate_n_members(members, i, newX, newy)
        print(f'{i} Ensemble R2s: {ensemble_score}')
        print(f'{i} Ensemble pred var: {ensemble_variance}')
        print(' ')

    Y_pred_train, var = ensemble_predictions(members, X_train)
    Y_pred_train = scaler.inverse_transform(Y_pred_train)
    Y_pred_test, var = ensemble_predictions(members, newX)
    Y_pred_test = scaler.inverse_transform(Y_pred_test)
    Y_train = scaler.inverse_transform(Y_train)
    Y_test = scaler.inverse_transform(newy)

if __name__ == "__main__":
    main()

    

