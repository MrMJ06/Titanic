import pandas as pd
from sklearn import tree
import numpy as np
import graphviz
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


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

    depth_val = np.arange(2, 11)
    leaf_val = np.arange(1, 31, 1)
    grid_s = [{'max_depth': depth_val, 'min_samples_leaf': leaf_val}]
    X_train, X_test, y_train, y_test = train_test_split(train_data, labels)

    model = tree.DecisionTreeClassifier(criterion='entropy')

    cv_tree = GridSearchCV(estimator=model, param_grid=grid_s, cv=StratifiedKFold(n_splits=10))
    cv_tree.fit(X_train, y_train)

    best_depth = cv_tree.best_params_['max_depth']
    best_leaf = cv_tree.best_params_['min_samples_leaf']

    split = StratifiedShuffleSplit(train_size=0.2, n_splits=1)
    print('Best depth is '+str(best_depth)+ ' and the best min number in leaf is '+ str(best_leaf))
    clf = tree.DecisionTreeClassifier(max_depth=best_depth, min_samples_leaf=best_leaf)
    best_clf = None

    for train_index, test_index in split.split(train_data, labels):
        X_train, y_train = np.array(train_data)[train_index],  np.array(labels)[train_index]
        X_test, y_test = np.array(train_data)[test_index],  np.array(labels)[test_index]


        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        print(classification_report(y_test, y_pred))
        best_clf = clf

    # dot_data = tree.export_graphviz(best_clf, out_file=None)
    # graph = graphviz.Source(dot_data)
    # graph.render("titanic")
    # print()
    # dot_data = tree.export_graphviz(best_clf, out_file=None,
    #                                 feature_names=list(train_data.columns.values),
    #                                 class_names=["Die", "Survive"],
    #                                 filled=True, rounded=True,
    #                                 special_characters=True)

    # graph = graphviz.Source(dot_data)
    # graph.view("Graph")

    test_data = pd.read_csv("resources/test.csv")
    test_data = test_data.fillna(train_data.mean())
    labels = test_data["PassengerId"]
    test_data_norm = preprocess_data(test_data, False)

    predictions = best_clf.predict(test_data_norm)
    dictionary = {'PassengerId': labels, 'Survived': predictions}
    result = pd.DataFrame(data=dictionary)
    # print(result.describe())

    # result[result.columns].to_csv(path_or_buf="results.csv", index=False)

    return result

__main__()