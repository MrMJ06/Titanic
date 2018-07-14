import pandas as pd
from sklearn import neural_network
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


# -----------------------------   Functions -------------------------------------


def preprocess_data(data):

    data.__delitem__("Cabin")
    data.__delitem__("PassengerId")
    data.__delitem__("Name")
    data.__delitem__("Ticket")
    data.__delitem__("Age")
    data.__delitem__("SibSp")
    data.__delitem__("Parch")
    data = data.fillna(data.mean())

    pclass_converted = pd.get_dummies(data["Pclass"], prefix="is")
    sex_converted = pd.get_dummies(data["Sex"], prefix="is")

    for pclass in pclass_converted:
        data[pclass] = pclass_converted[pclass]

    for sex in sex_converted:
        data[sex] = sex_converted[sex]

    data.__delitem__("Pclass")
    data.__delitem__("Embarked")
    data.__delitem__("Sex")
    # data = shuffle(train_data)

    # data = train_data.dropna()

    data_norm = (data - data.min()) / (data.max() - data.min())
    np.asarray(data, np.float64)

    return data_norm


def train(data_norm, data_label, it):
    best_accuracy = 0
    best_classifier = None

    pd.set_option("display.max_rows", 500)
    for i in range(0, it):
        train_data_input = data_norm

        # data_train, data_test, labels_train, labels_test = train_test_split(train_data_input, train_data_label, test_size=0.20, stratify=train_data_label)
        classifier = neural_network.MLPClassifier(hidden_layer_sizes=[10, 5], max_iter=1000, learning_rate_init=0.001)

        kf = StratifiedKFold(n_splits=10, shuffle=True)
        mean_accuracy = 0
        partitions = kf.split(train_data_input, data_label)
        for train_index, test_index in partitions:
            X_train, X_test = np.array(train_data_input)[train_index], np.array(train_data_input)[test_index]
            y_train, y_test = np.array(data_label)[train_index], np.array(data_label)[test_index]

            classifier.fit(X_train, y_train)

            predictions = classifier.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            mean_accuracy += accuracy / 10

        if best_accuracy < mean_accuracy:
            best_accuracy = mean_accuracy
            best_classifier = classifier
            print('Iteration: ' + str(i) + ' ' + str(best_accuracy))
    return best_classifier


# -----------------------------------------  main -------------------------------------


def __main__():

    train_data = pd.read_csv("resources/train.csv")
    train_data.dropna()
    train_data_norm = preprocess_data(train_data)
    train_data_label = train_data['Survived']
    train_data_norm.__delitem__("Survived")

    test_data = pd.read_csv("resources/test.csv")

    labels = test_data["PassengerId"]
    test_data_norm = preprocess_data(test_data)
    print(train_data_norm.describe())
    print(test_data_norm.describe())
    classifier = train(train_data_norm, train_data_label, 50)

    predictions = classifier.predict(test_data_norm)
    dictionary = {'PassengerId': labels, 'Survived': predictions}
    result = pd.DataFrame(data=dictionary)
    print(result.describe())

    result[result.columns].to_csv(path_or_buf="results.csv", index=False)


__main__()