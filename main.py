import pandas as pd
import KNNImplementation
import DecissionTreeImplemetation
import LogisticRegressorImplementation
import NaiveBayesImplementation
import neuralNetworkImplementation
# -----------------------------------------  main -------------------------------------


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

    # data = shuffle(train_data)

    # data = train_data.dropna()

    data_norm = (data - data.min()) / (data.max() - data.min())
    print(data_norm[1:8])

    return data_norm

def __main__():

    tree_result = DecissionTreeImplemetation.__main__()
    knn_result = KNNImplementation.__main__()
    logistic_result = LogisticRegressorImplementation.__main__()
    naive_result = NaiveBayesImplementation.__main__()
    neural_result = neuralNetworkImplementation.__main__()

    total_result = tree_result["Survived"] + knn_result["Survived"] + logistic_result["Survived"] + naive_result["Survived"] + neural_result["Survived"]
    total_result = total_result > 2
    total_result = total_result.astype(int)

    print(total_result)

    test_data = pd.read_csv("resources/test.csv")

    labels = test_data["PassengerId"]

    dictionary = {'PassengerId': labels, 'Survived': total_result}
    result = pd.DataFrame(data=dictionary)

    print(result)

    result[result.columns].to_csv(path_or_buf="results.csv", index=False)


__main__()