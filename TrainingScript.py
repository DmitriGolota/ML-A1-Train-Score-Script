"""
Dmitri Golota
A00941422
"""


import numpy as np
import pandas as pd
import statsmodels.api as sm
import keras
from keras import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import warnings
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from keras.layers import Dropout
from pickle import dump
from keras.optimizers import Adam
from numpy import dstack


PATH = '/Users/dimagolota/Documents/Term 4/COMP 4948 - Machine Learning/Assignments/ds_salaries.csv'

def train():

    batch_size = 0
    epochs = 0

    tf.config.list_physical_devices('GPU')
    warnings.simplefilter(action="ignore", category=FutureWarning)

    # Show all columns.
    pd.set_option('display.max_columns', None)

    # Increase number of columns that display on one line.
    pd.set_option('display.width', 1000)

    # load the dataset
    df = pd.read_csv(PATH, sep=',')
    # split into input (X) and output (y) variables
    max_year = np.max(df["work_year"])
    # set values to simple integers
    df["work_year"] = [max_year - e for e in df["work_year"]]
    # clip out higher end outliers
    df["salary_in_usd"] = np.clip(df["salary_in_usd"], a_min=0, a_max=300000)
    print(df.describe())

    # Show data in plot by non-integer values
    sns.scatterplot(data=df, x="employment_type", y="salary_in_usd")
    plt.show()
    sns.scatterplot(data=df, x="experience_level", y="salary_in_usd")
    plt.show()
    sns.set(rc={'figure.figsize': (20, 8)})
    plt.xticks(rotation=90)
    sns.scatterplot(data=df, x="job_title", y="salary_in_usd")
    plt.show()

    # Treat Data:
    x = df[["work_year", "experience_level", "employment_type", "job_title", "employee_residence", "remote_ratio",
            "company_location", "company_size"]]
    x = pd.get_dummies(x, columns=["experience_level", "employment_type", "job_title", "remote_ratio",
                                   "employee_residence", "company_location", "company_size"])

    # Keep significant features
    x = x[['job_title_Data Analyst', 'experience_level_SE', 'experience_level_EN', 'experience_level_EX', 'experience_level_MI',
         'employee_residence_US', 'company_location_CA', 'company_location_GB']]
    y = df[['salary_in_usd']]

    print(x, y)

    # Scale Data: 80/20 train-test split
    unscaled_train_x, unscaled_test_x, unscaled_train_y, unscaled_test_y = train_test_split(x, y, test_size=0.2)

    x_scaler = StandardScaler()
    x_scaler.fit(unscaled_train_x)
    train_x = x_scaler.transform(unscaled_train_x)
    test_x = x_scaler.transform(unscaled_test_x)

    y_scaler = StandardScaler()
    y_scaler.fit(unscaled_train_y)
    train_y = y_scaler.transform(unscaled_train_y)
    test_y = y_scaler.transform(unscaled_test_y)


    # OLS MODEL:
    def build_ols_model(train_x, test_x, train_y, test_y):
        model = sm.OLS(train_y, train_x).fit()
        predictions = model.predict(test_x)
        print(model.summary())
        unscaled_pred = y_scaler.inverse_transform(predictions)
        print('Root Mean Squared Error:',
              np.sqrt(metrics.mean_squared_error(unscaled_test_y, unscaled_pred)))

    build_ols_model(train_x, test_x, train_y, test_y)

    # Define the model.
    def create_model():
        model = Sequential()
        model.add(Dense(8, input_dim=8, kernel_initializer='nor'
                                                           'mal', activation='softsign'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='linear'))
        sgd = tf.optimizers.SGD(learning_rate=0.01, momentum=0.9, clipnorm=1.0)
        # opt = Adam(lr=0.01)
        model.compile(loss='mean_squared_error', optimizer=sgd)
        return model

    ### Grid Building Section #######################
    estimator = KerasRegressor(build_fn=create_model)
    # define the grid search parameters
    batch_size = [4, 8, 16, 32]
    epochs = [25, 50, 100, 200, 500]

    param_grid = dict(batch_size=batch_size, epochs=epochs)
    grid = GridSearchCV(estimator=estimator, param_grid=param_grid, n_jobs=-1, cv=3, verbose=0)
    #################################################
    grid_result = grid.fit(train_x, train_y)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


    ## Optimizating Optimizer
    def create_model_optimizer(optimizer):
        model = Sequential()
        model.add(Dense(8, input_dim=8, kernel_initializer='normal', activation='softsign'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model

    ### Grid Building Section #######################
    estimator = KerasRegressor(build_fn=create_model_optimizer, epochs=25, batch_size=16, verbose=0)

    # Define the grid search parameters.
    optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    param_grid = dict(optimizer=optimizer)
    grid = GridSearchCV(estimator=estimator, param_grid=param_grid, n_jobs=-1, cv=3)
    #################################################
    grid_result = grid.fit(train_x, train_y)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))



    ## Optimizing Learning Rate
    def create_model_learning_rate(learning_rate):
        model = Sequential()
        model.add(Dense(8, input_dim=8, kernel_initializer='normal', activation='softsign'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='linear'))
        adam = keras.optimizers.RMSprop(learning_rate=learning_rate)
        model.compile(loss='mean_squared_error', optimizer=adam)
        return model

    ### Grid Building Section #######################
    estimator = KerasRegressor(build_fn=create_model_learning_rate, epochs=25, batch_size=16, verbose=0)

    # Define the grid search parameters.
    learning_rate = [0.001, 0.01, 0.1, 0.2, 0.5]
    param_grid = dict(learning_rate=learning_rate)
    grid = GridSearchCV(estimator=estimator, param_grid=param_grid, n_jobs=-1, cv=3)
    #################################################
    grid_result = grid.fit(train_x, train_y)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


    ## Optimize KErnal Initializer
    def create_model_kernel(kernel_init):
        model = Sequential()
        model.add(Dense(8, input_dim=8, kernel_initializer=kernel_init, activation='softsign'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='linear'))
        adam = tf.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mean_squared_error', optimizer=adam)
        return model

    ### Grid Building Section #######################
    estimator = KerasRegressor(build_fn=create_model_kernel, epochs=25, batch_size=16, verbose=0)

    # Define the grid search parameters.
    init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal',
                 'he_uniform']
    param_grid = dict(kernel_init=init_mode)
    grid = GridSearchCV(estimator=estimator, param_grid=param_grid, n_jobs=-1, cv=3)
    #################################################
    grid_result = grid.fit(train_x, train_y)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


    ## Optimize Activation Function
    def create_model_activation(activation_func):
        model = Sequential()
        model.add(Dense(8, input_dim=8, kernel_initializer='uniform', activation=activation_func))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='linear'))
        adam = tf.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mean_squared_error', optimizer=adam)
        return model

    ### Grid Building Section #######################
    estimator = KerasRegressor(build_fn=create_model_activation, epochs=25, batch_size=16, verbose=0)

    # Define the grid search parameters.
    funcs = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    param_grid = dict(activation_func=funcs)
    grid = GridSearchCV(estimator=estimator, param_grid=param_grid, n_jobs=-1, cv=3)
    #################################################
    grid_result = grid.fit(train_x, train_y)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


    ## Optimize number of nuerons
    def create_model_num_neurons(num_neurons):
        model = Sequential()
        model.add(Dense(num_neurons, input_dim=8, kernel_initializer='uniform', activation='linear'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='linear'))
        adam = tf.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mean_squared_error', optimizer=adam)
        return model

    ### Grid Building Section #######################
    estimator = KerasRegressor(build_fn=create_model_num_neurons, epochs=25, batch_size=16, verbose=0)

    # Define the grid search parameters.
    neurons = [5, 10, 20, 100, 150, 200]
    param_grid = dict(num_neurons=neurons)
    grid = GridSearchCV(estimator=estimator, param_grid=param_grid, n_jobs=-1, cv=3)
    #################################################
    grid_result = grid.fit(train_x, train_y)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))



    ## Create Ultimate Model
    # Since this is a linear regression use KerasRegressor.
    def create_final_model():
        model = Sequential()
        model.add(Dense(200, input_dim=8, kernel_initializer='uniform', activation='linear'))
        model.add(Dense(16, input_dim=8, kernel_initializer='uniform', activation='linear'))
        model.add(Dense(8, input_dim=8, kernel_initializer='uniform', activation='linear'))
        model.add(Dense(2, input_dim=8, kernel_initializer='uniform', activation='linear'))
        # Output layer
        model.add(Dense(1, activation='linear'))
        RMSprop = tf.optimizers.RMSprop(learning_rate=0.01)
        model.compile(loss='mean_squared_error', optimizer=RMSprop)
        return model

    model = create_final_model()
    pred = model.predict(test_x)
    unscaled_pred = y_scaler.inverse_transform(pred)
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(unscaled_test_y, unscaled_pred)))


    ## Save Model

    # save the scaler
    dump(x_scaler, open('Binaries/x_scaler.pkl', 'wb'))
    dump(y_scaler, open('Binaries/y_scaler.pkl', 'wb'))

    # fit and save models
    numModels = 5

    members = list()
    print("\nFitting models with training data.")
    for i in range(numModels):
        # fit model
        model = create_final_model()
        model.fit(train_x, train_y, verbose=0, batch_size=16, epochs=50)
        filename = 'models' + "/" + 'model_' + str(i + 1) + '.h5'
        model.save(filename)
        members.append(model)
        print('>Saved %s' % filename)

    # fit and save stacked model
    # create stacked model input dataset as outputs from the ensemble
    def stacked_dataset(members, inputX):
        stackX = None
        for model in members:
            # make prediction
            yhat = model.predict(inputX, verbose=0)
            # stack predictions into [rows, members, probabilities]
            if stackX is None:
                stackX = yhat
            else:
                stackX = dstack((stackX, yhat))
        # flatten predictions to [rows, members x probabilities]
        stackX = stackX.reshape((stackX.shape[0], stackX.shape[1] * stackX.shape[2]))
        return stackX

    # fit a model based on the outputs from the ensemble members
    def fit_stacked_model(members, inputX, inputy):
        # create dataset using ensemble
        stackedX = stacked_dataset(members, inputX)
        # fit standalone model
        model = sm.OLS(inputy, stackedX).fit()
        return model

    fitted_stacked_model = fit_stacked_model(members, test_x, test_y)
    dump(fitted_stacked_model, open('models/stacked_model.h5', 'wb'))

if __name__ == '__main__':
    train()