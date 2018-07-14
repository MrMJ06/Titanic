import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import numpy as np
import graphviz
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold

# -----------------------------------------  main -------------------------------------
#  TODO: Usar matriz de correspondencia

def preprocess_data(data, test):

    dicS = {"male": 0., "female": 1.}
    data.__delitem__("Embarked")
    data.__delitem__("Cabin")
    data.__delitem__("PassengerId")
    data.__delitem__("Parch")
    data.__delitem__("SibSp")
    data.__delitem__("Name")
    data.__delitem__("Age")

    if test:
        data.__delitem__("Survived")
    data.__delitem__("Ticket")
    # data = shuffle(train_data)
    data.replace(dicS, inplace=True)
    print(data[1:8])
    # data = train_data.dropna()

    # data_norm = (data - data.min()) / (data.max() - data.min())

    return data


def __main__():

    train_data = pd.read_csv("resources/train.csv")
    train_data.dropna()
    labels = train_data["Survived"]

    train_data = preprocess_data(train_data, True)

    pd.set_option('display.max_columns', 20)

    split = StratifiedShuffleSplit(labels, test_size=0.2, n_iter=1)

    best_clf = None

    for train_index, test_index in split:
        X_train, y_train = np.array(train_data)[train_index],  np.array(labels)[train_index]
        X_test, y_test = np.array(train_data)[test_index],  np.array(labels)[test_index]

        clf = tree.DecisionTreeClassifier(max_depth=3)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        print(classification_report(y_test, y_pred))
        best_clf = clf

    dot_data = tree.export_graphviz(best_clf, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("titanic")
    print()
    dot_data = tree.export_graphviz(best_clf, out_file=None,
                                    feature_names=list(train_data.columns.values),
                                    class_names=["Die", "Survive"],
                                    filled=True, rounded=True,
                                    special_characters=True)

    graph = graphviz.Source(dot_data)
    graph.view("Graph")

    test_data = pd.read_csv("resources/test.csv")
    test_data = test_data.fillna(train_data.mean())
    labels = test_data["PassengerId"]
    test_data_norm = preprocess_data(test_data, False)

    predictions = best_clf.predict(test_data_norm)
    dictionary = {'PassengerId': labels, 'Survived': predictions}
    result = pd.DataFrame(data=dictionary)
    print(result.describe())

    result[result.columns].to_csv(path_or_buf="results.csv", index=False)
__main__()