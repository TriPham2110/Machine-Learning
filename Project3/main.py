import logging
import traceback
import warnings

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from neuralnet import NeuralNet

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    try:
        print('Statlog (Heart) Data Set')
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
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=0)

        neuralnet = NeuralNet(num_input_nodes=13, num_output_nodes=1, hidden_layers=[10, 10], learning_rate=0.01)
        neuralnet.train(Xtrain, ytrain)

        train_pred = neuralnet.predict(Xtrain)
        test_pred = neuralnet.predict(Xtest)

        print("Train accuracy is {}".format(accuracy_score(ytrain.reshape(-1, ), train_pred)))
        print("Test accuracy is {}".format(accuracy_score(ytest.reshape(-1, ), test_pred)))

        neuralnet.plot_loss('Statlog (Heart) Data Set')

        print('Breast Cancer Wisconsin (Diagnostic) Data Set')
        dataset2 = pd.read_csv("data/breast_cancer_wisconsin_diagnostic_data.csv")
        dataset2.dropna(axis="columns", how="any", inplace=True)

        dataset2['diagnosis'] = dataset2['diagnosis'].replace('B', 0)
        dataset2['diagnosis'] = dataset2['diagnosis'].replace('M', 1)

        X2 = dataset2.drop(columns=['id', 'diagnosis'])
        y2 = dataset2['diagnosis'].values.reshape(X2.shape[0], 1)
        Xtrain2, Xtest2, ytrain2, ytest2 = train_test_split(X2, y2, test_size=0.2, random_state=0)

        neuralnet2 = NeuralNet(num_input_nodes=30, num_output_nodes=1, hidden_layers=[25, 20, 15], learning_rate=0.1)
        neuralnet2.train(Xtrain2, ytrain2)

        train_pred2 = neuralnet2.predict(Xtrain2)
        test_pred2 = neuralnet2.predict(Xtest2)

        print("Train accuracy is {}".format(accuracy_score(ytrain2.reshape(-1, ), train_pred2)))
        print("Test accuracy is {}".format(accuracy_score(ytest2.reshape(-1, ), test_pred2)))

        neuralnet2.plot_loss('Breast Cancer Wisconsin (Diagnostic) Data Set')
    except Exception as e:
        logging.error(traceback.format_exc())
