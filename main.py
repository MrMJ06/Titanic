import pandas as pd
from sklearn import neural_network
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold

# -----------------------------------------  main -------------------------------------
#  TODO: Usar matriz de correspondencia

def preprocess_data(data):

    data = data.fillna(data.mean())

    data.__delitem__("Cabin")
    data.__delitem__("PassengerId")
    data.__delitem__("Name")
    data.__delitem__("Ticket")
    # data = shuffle(train_data)
    print(data[1:8])
    # data = train_data.dropna()

    #data_norm = (data - data.min()) / (data.max() - data.min())

    return data

def __main__():

    train_data = pd.read_csv("resources/train.csv")
    train_data = preprocess_data(train_data)
    pd.set_option('display.max_columns', 20)
    print(train_data.corr()["Survived"])
    print(train_data.describe())
    train_data.to_csv(path_or_buf="process_train.csv", index=False)

__main__()