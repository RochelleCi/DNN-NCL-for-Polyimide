import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, resample
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Define the relative paths to the data files
data_dir = os.path.join(os.path.dirname(__file__), 'data')
dataset_path = os.path.join(data_dir, 'Dataset.csv')
fing_path = os.path.join(data_dir, 'Fing.csv')

# Custom nanmean_squared_error loss function
def nanmean_squared_error(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    residuals = y_true - y_pred
    residuals_no_nan = tf.where(tf.math.is_nan(residuals), tf.zeros_like(residuals), residuals)
    sum_residuals = tf.reduce_sum(tf.square(residuals_no_nan), -1) / tf.reduce_sum(tf.cast(~tf.math.is_nan(y_true), tf.float64), -1)
    return sum_residuals

# Evaluate a single DNN for regression 
def evaluate_model(trainX, trainy, testX, testy):
    # Define model
    model = Sequential()
    model.add(Dense(units=128, activation='relu', input_shape=(trainX.shape[1],)))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=6))

    model.compile(loss=nanmean_squared_error, optimizer='adam')
    history = model.fit(trainX, trainy, epochs=200, batch_size=64, validation_data=(testX, testy), verbose=0)
    
    # Evaluate the model
    predy = model.predict(testX)
    test_r2 = np.zeros(6)
    for i in range(6):
        flag1 = ~np.isnan(testy[:,i])
        test_r2[i] = r2_score(testy[:,i][flag1], predy[:,i][flag1])
    
    return model, test_r2

# Make an ensemble prediction
def ensemble_predictions(members, testX):
    # Make predictions
    yhats = [model.predict(testX) for model in members]
    yhats = np.array(yhats)
    # Mean of ensemble members
    predictions = np.mean(yhats, axis=0)
    variances = np.var(yhats, axis=0)
    
    return predictions, variances

# Evaluate a specific number of members in an ensemble for the regression score (R2) and the variance of predictions
def evaluate_n_members(members, n_members, testX, testy):
    # Select a subset of members
    subset = members[:n_members]
    # Make prediction
    yhat, variances = ensemble_predictions(subset, testX)
    avg_var = np.mean(variances, axis=0)
    
    # Calculate R2
    test_r2 = np.zeros(6)
    for i in range(6):
        flag1 = ~np.isnan(testy[:,i])
        test_r2[i] = r2_score(testy[:,i][flag1], yhat[:,i][flag1])
    
    return test_r2, avg_var

# Read and process the training data
Dataset_Smiles = pd.read_csv(dataset_path)
Dataset_Grouped = Dataset_Smiles.groupby('Smiles').mean().reset_index()

Y = Dataset_Grouped.iloc[:,-6:-1]

# Normalize Y
Y = np.array(Y)
scaler = StandardScaler()
Y = scaler.fit_transform(Y)

X = pd.read_csv(fing_path)

# Normalize X
X = np.array(X)
Xscaler = StandardScaler()
X = Xscaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

# Initialize variables for ensemble learning
n_splits = 16
scores, members = list(), list()

for _ in range(n_splits):
    # Select indexes
    ix = [i for i in range(len(X))]
    train_ix = resample(ix, replace=True, n_samples=round(X.shape[0] * 0.8)) # Bootstrap 80% of the training set for each model
    test_ix = [x for x in ix if x not in train_ix]
    
    # Select data
    trainX, trainy = X[train_ix], Y[train_ix]
    testX, testy = X[test_ix], Y[test_ix]
    
    # Evaluate model
    model, test_r2 = evaluate_model(trainX, trainy, testX, testy)
    print(test_r2)
    scores.append(test_r2)
    members.append(model)

# Summarize expected performance
print('-----------------------------------------')
print('Expected R2: %.3f' % np.mean(scores))
print('Mean R2: ' + str(np.mean(scores, axis=0)))
print('Estimated Std of R2: ' + str(np.std(scores, axis=0)))
print('-----------------------------------------')

# Evaluate different numbers of ensembles on hold out set
newX, newy = X_test, Y_test

for i in range(1, n_splits + 1):
    ensemble_score, ensemble_variance = evaluate_n_members(members, i, newX, newy)
    print(f'{i} Ensemble R2s: {ensemble_score}')
    print(f'{i} Ensemble pred var: {ensemble_variance}')
    print(' ')

Y_pred_train, var_train = ensemble_predictions(members, X_train)
Y_pred_train = scaler.inverse_transform(Y_pred_train)
Y_pred_test, var_test = ensemble_predictions(members, newX)
Y_pred_test = scaler.inverse_transform(Y_pred_test)
Y_train = scaler.inverse_transform(Y_train)
Y_test = scaler.inverse_transform(newy)

# Function to calculate MSE and MAE for each of the six prediction sets
def calculate_metrics(Y_true, Y_pred):
    mse_scores = []
    mae_scores = []

    for i in range(6):
        # Filtering out NaN values from both predictions and true values
        mask = ~np.isnan(Y_true[:, i])
        mse = mean_squared_error(Y_true[mask, i], Y_pred[mask, i])
        mae = mean_absolute_error(Y_true[mask, i], Y_pred[mask, i])

        mse_scores.append(mse)
        mae_scores.append(mae)
    
    return mse_scores, mae_scores

# Calculate metrics for the training set
mse_scores_train, mae_scores_train = calculate_metrics(Y_train, Y_pred_train)

# Calculate metrics for the testing set
mse_scores_test, mae_scores_test = calculate_metrics(Y_test, Y_pred_test)

# Print the MSE and MAE for each set for training and testing datasets
for i in range(6):
    print(f"Output {i+1} - Training: MSE = {mse_scores_train[i]}, MAE = {mae_scores_train[i]}")
    print(f"Output {i+1} - Testing: MSE = {mse_scores_test[i]}, MAE = {mae_scores_test[i]}")
