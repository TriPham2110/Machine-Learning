import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from neuralnet import NeuralNet
import logging
import traceback
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    try:
        columns = ['age', 'sex', 'chest_pain', 'resting_blood_pressure',
                   'serum_cholestoral', 'fasting_blood_sugar', 'resting_ecg_results',
                   'max_heart_rate_achieved', 'exercise_induced_angina', 'oldpeak', "slope of the peak",
                   'num_of_major_vessels', 'thal', 'heart_disease']
        dataset = pd.read_csv("data/heart.dat", sep=' ', names=columns)
        dataset.dropna(axis="columns", how="any", inplace=True)

        dataset['heart_disease'] = dataset['heart_disease'].replace(1, 0)
        dataset['heart_disease'] = dataset['heart_disease'].replace(2, 1)

        X = dataset.drop(columns=['heart_disease'])
        y = dataset['heart_disease'].values.reshape(X.shape[0], 1)
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=2)

        neuralnet = NeuralNet(num_input_nodes=13, num_output_nodes=1, hidden_layers=[10, 10], learning_rate=0.1)
        neuralnet.train(Xtrain, ytrain)

        test_pred = neuralnet.predict(Xtest)

        print(ytest.reshape(-1,))
        print(test_pred)

    except Exception as e:
        logging.error(traceback.format_exc())
