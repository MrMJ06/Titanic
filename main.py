import pandas as pd

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

    train_data = pd.read_csv("resources/train.csv")
    train_data = preprocess_data(train_data)
    pd.set_option('display.max_columns', 20)

    print(train_data.corr()["Survived"])
    print(train_data.describe())

    train_data.to_csv(path_or_buf="process_train.csv", index=False)

__main__()