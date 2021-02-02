import argparse

# importing necessary libraries
import os

import joblib
import numpy as np
import pandas as pd
from azureml.core.run import Run
from sklearn.metrics import confusion_matrix

run = Run.get_context()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type=float, default=1.0,
                        help='Learning Rate to be used in the algorithm')
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='Number of Trees to be used in the algorithm')

    args = parser.parse_args()
    learning_rate = np.float(args.learning_rate)
    n_estimators = np.int(args.n_estimators)
    run.log('Learning Rate', learning_rate)
    run.log('Number of Estimators', n_estimators)
    # loading the iris dataset
    training_dataset_path = 'flight-delays-train-dataset.csv'
    validation_dataset_path = 'flight-delays-validation-dataset.csv'
    train_ds = pd.read_csv(filepath_or_buffer=training_dataset_path,
                           header=0)
    validation_ds = pd.read_csv(filepath_or_buffer=validation_dataset_path,
                                header=0)

    # X -> features, y -> label
    X_train = train_ds.loc[:, (train_ds.columns != 'Cancelled') & (train_ds.columns != 'Carrier')]
    X_val = validation_ds.loc[:, (validation_ds.columns != 'Cancelled') & (validation_ds.columns != 'Carrier')]
    y_train = train_ds.loc[:, train_ds.columns == 'Cancelled']
    y_val = validation_ds.loc[:, validation_ds.columns == 'Cancelled']
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                     max_depth=1, random_state=0).fit(X_train, y_train.to_numpy().ravel())
    predictions = clf.predict(X_val)
    # model accuracy for X_test
    accuracy = clf.score(X_val, y_val)
    print('Accuracy of SVM classifier on validation set: {:.2f}'.format(accuracy))
    run.log('Accuracy', np.float(accuracy))
    cm = confusion_matrix(y_val.to_numpy().ravel(), predictions)
    print(cm)

    os.makedirs('outputs', exist_ok=True)
    # files saved in the "outputs" folder are automatically uploaded into run history
    joblib.dump(clf, 'outputs/model.joblib')


if __name__ == '__main__':
    main()
