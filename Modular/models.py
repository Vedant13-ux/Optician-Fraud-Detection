import tensorflow as tf
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier



def svm_classifier(X_train: np.ndarray, y_train : np.ndarray, kernel: str, gamma: int):
    classifier = SVC(kernel = kernel, random_state = 0, gamma = gamma)
    classifier.fit(X_train, y_train)
    return classifier

def random_forest_classifier(X_train: np.ndarray, y_train : np.ndarray, n_estimators: int, criterion: str ):
    classifier = RandomForestClassifier(n_estimators = n_estimators, criterion = criterion, random_state = 0)
    classifier.fit(X_train, y_train)
    return classifier

def xgboost_classifier(X_train: np.ndarray, y_train : np.ndarray):
    classifier = XGBClassifier()
    classifier.fit(X_train, y_train)

def ann_classifier(X_train: np.ndarray, y_train : np.ndarray, metrics: list, optimizer: str ,epochs: int, batch_size: int ):
    classifier = tf.keras.models.Sequential()
    classifier.add(tf.keras.layers.Dense(units=6, activation='relu'))
    classifier.add(tf.keras.layers.Dense(units=6, activation='relu'))
    classifier.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    # Compiling the classifier
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = metrics)

    # Training the classifier on the Training set
    classifier.fit(X_train, y_train, batch_size = batch_size, epochs = epochs)

    return classifier


def predict_results(classifier , X_test: np.ndarray, ann: bool):
    y_pred = classifier.predict(X_test)
    if ann:
        y_pred = np.where(y_pred > 0.5, 1, 0)

    return y_pred