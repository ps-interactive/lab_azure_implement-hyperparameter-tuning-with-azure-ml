import argparse

# importing necessary libraries
import numpy as np
from azureml.core import Dataset, Workspace
from azureml.core.run import Run

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
    ws = Workspace.from_config()
    # loading the iris dataset
    training_dataset_name = 'flight-delays-train-dataset'
    validation_dataset_name = 'flight-delays-validation-dataset'
    train_ds = Dataset.get_by_name(workspace=ws, name=training_dataset_name).to_pandas_dataframe()
    validation_ds = Dataset.get_by_name(workspace=ws, name=validation_dataset_name).to_pandas_dataframe()

    # X -> features, y -> label
    X_train = train_ds.loc[:, train_ds.columns != 'Cancelled']
    X_val = validation_ds.loc[:, validation_ds.columns != 'Cancelled']
    y_train = train_ds.Cancelled
    y_val = validation_ds.Cancelled

    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                     max_depth=1, random_state=0).fit(X_train, y_train)

    # model accuracy for X_test
    accuracy = clf.score(X_val, y_val)
    print('Accuracy of SVM classifier on validation set: {:.2f}'.format(accuracy))
    run.log('Accuracy', np.float(accuracy))


if __name__ == '__main__':
    main()
