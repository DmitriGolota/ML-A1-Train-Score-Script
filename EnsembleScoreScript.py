"""
Dmitri Golota
A00941422
"""


import numpy as np
import pandas as pd
from keras.models import load_model
import pickle
from numpy import dstack

PATH = '/Users/dimagolota/Documents/Term 4/COMP 4948 - Machine Learning/Assignments/ds_salaries_mystery.csv'

def score():


    def load_mystery_data(x_scaler):
        data = pd.read_csv(PATH, sep=',')
        data = data.drop(columns=['company_size'])
        max_year = np.max(data["work_year"])
        data["work_year"] = [max_year - year for year in data["work_year"]]
        data = pd.get_dummies(data)
        features = ['experience_level_EN', 'experience_level_EX',
                    'experience_level_MI', 'experience_level_SE',
                    'job_title_Data Analyst', 'employee_residence_US',
                 'company_location_CA', 'company_location_GB']

        for feature in features:
            if feature not in data.columns:
                data[feature] = 0
        data = data[features]
        print(data.describe())
        X_scale = x_scaler.transform(data)

        return data, X_scale

    # load models from file
    def load_all_models(n_models):
        all_models = list()
        for i in range(n_models):
            # define filename for this ensemble
            filename = 'models/model_' + str(i + 1) + '.h5'
            # load model from file
            model = load_model(filename)
            # add to list of models
            all_models.append(model)
            print('>loaded %s' % filename)
        return all_models

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

    # Make predictions with the stacked model
    def stacked_prediction(members, stackedModel, inputX):
        # create dataset using ensemble
        stackedX = stacked_dataset(members, inputX)
        # make a prediction
        yhat = stackedModel.predict(stackedX)
        return yhat

    y_scaler = pickle.load(open('Binaries/y_scaler.pkl', 'rb'))
    x_scaler = pickle.load(open('Binaries/x_scaler.pkl', 'rb'))

    X, X_scale = load_mystery_data(x_scaler)

    numModels = 5
    members = load_all_models(numModels)
    print('Loaded %d models' % len(members))

    stacked_model = pickle.load(open('models/stacked_model.h5', 'rb'))

    yhat = stacked_prediction(members, stacked_model, X_scale)
    yhat = y_scaler.inverse_transform(yhat)
    print(yhat)
    result = pd.DataFrame()
    result['salary_in_usd'] = yhat.tolist()
    print(result)
    result.to_csv('ds_salaries_predictions.csv', index=False)


if __name__ == '__main__':
    score()