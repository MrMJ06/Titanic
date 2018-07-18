import pandas as pd
from sklearn import naive_bayes
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

# -----------------------------------------  main -------------------------------------
#  TODO: Usar matriz de correspondencia

def preprocess_data(data):

    data = data.fillna(data.mean())
    data.__delitem__("Cabin")
    data.__delitem__("PassengerId")
    data.__delitem__("Name")
    data.__delitem__("Ticket")
    data.__delitem__("Age")
    data.__delitem__("SibSp")
    data.__delitem__("Parch")

    pclass_converted = pd.get_dummies(data["Pclass"], prefix="is")
    sex_converted = pd.get_dummies(data["Sex"], prefix="is")

    for pclass in pclass_converted:
        data[pclass] = pclass_converted[pclass]

    for sex in sex_converted:
        data[sex] = sex_converted[sex]

    data.__delitem__("Pclass")
    data.__delitem__("Embarked")
    data.__delitem__("Sex")

    data_norm = (data - data.min()) / (data.max() - data.min())
    print(data_norm[1:8])

    return data_norm

def __main__():

    train_data = pd.read_csv("resources/train.csv")
    train_data.dropna()
    labels = train_data['Survived']
    train_data.__delitem__('Survived')
    train_data = preprocess_data(train_data)
    pd.set_option('display.max_columns', 20)

    best_cls = naive_bayes.MultinomialNB()

    kfold = StratifiedKFold(n_splits=10)
    split = kfold.split(train_data, labels)
    mean_accuracy = 0

    for train_index, test_index in split:
        X_train, X_test = np.array(train_data)[train_index], np.array(train_data)[test_index]
        y_train, y_test = np.array(labels)[train_index], np.array(labels)[test_index]
        best_cls.fit(X_train, y_train)

        predictions = best_cls.predict(X_test)
        mean_accuracy += accuracy_score(y_test, predictions) / 10
        print(confusion_matrix(y_test, predictions))

    print('mean accuracy: ' + str(mean_accuracy))

    test_data = pd.read_csv("resources/test.csv")

    ids = test_data["PassengerId"]
    test_data_norm = preprocess_data(test_data)

    results = best_cls.predict(test_data_norm)

    dictionary = {'PassengerId': ids, 'Survived': results}
    result = pd.DataFrame(data=dictionary)
    print(result.describe())

    result[result.columns].to_csv(path_or_buf="results.csv", index=False)

__main__()