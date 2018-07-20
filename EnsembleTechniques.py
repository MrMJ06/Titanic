from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble.forest import (RandomForestClassifier, ExtraTreesClassifier)
import pandas as pd


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

    return data_norm


train_data = pd.read_csv('resources/train.csv')
train_data = train_data.dropna()
train_data = preprocess_data(train_data)

X = train_data[['is_1', 'is_2', 'is_3', 'Fare', 'is_male', 'is_female']]
Y = train_data['Survived']

XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.2)

n_estimators = 100

models = [DecisionTreeClassifier(max_depth=3), BaggingClassifier(n_estimators=n_estimators),
          RandomForestClassifier(n_estimators=n_estimators), ExtraTreesClassifier(n_estimators=n_estimators),
          AdaBoostClassifier(n_estimators=n_estimators)]

model_title = ['DecisionTree', 'Bagging', 'RandomForest', 'ExtraTrees', 'AdaBoost']

surv_preds, surv_probs, scores, fprs, tprs, thres = ([] for i in range(6))

for i, model in enumerate(models):
    print('Fitting {0}'.format(model_title[i]))

    clf = model.fit(XTrain, YTrain)
    surv_preds.append(model.predict(XTest))
    surv_probs.append(model.predict_proba(XTest))
    scores.append(model.score(XTest, YTest))

    fpr, tpr, thresholds = roc_curve(YTest, surv_probs[i][:, 1])
    fprs.append(fpr)
    tprs.append(tpr)
    thres.append(thresholds)

for i, score in enumerate(scores):
    print('{0} with score {1:0.2f}'.format(model_title[i], score))
