import pandas as pd
from sklearn import neural_network
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split

train_data = pd.read_csv("resources/train.csv")
train_data.__delitem__("Cabin")
train_data.__delitem__("PassengerId")
train_data.__delitem__("Age")
train_data.__delitem__("Name")
train_data.__delitem__("Ticket")
train_data.__delitem__("SibSp")
train_data.__delitem__("Parch")


pclass_converted = pd.get_dummies(train_data["Pclass"], prefix="is")
embarked_converted = pd.get_dummies(train_data["Embarked"], prefix="is")
sex_converted = pd.get_dummies(train_data["Sex"], prefix="is")

for pclass in pclass_converted:
    train_data[pclass] = pclass_converted[pclass]

for embark in embarked_converted:
    train_data[embark] = embarked_converted[embark]

for sex in sex_converted:
    train_data[sex] = sex_converted[sex]

# train_data.__delitem__("Pclass")
train_data.__delitem__("Embarked")
train_data.__delitem__("Sex")
# train_data = shuffle(train_data)

train_data = train_data.dropna()

train_data_norm = (train_data - train_data.mean()) / (train_data.max() - train_data.min())
np.asarray(train_data_norm, np.float64)


test_data = pd.read_csv("resources/test.csv")
test_data.__delitem__("Cabin")
test_data.__delitem__("Name")
test_data.__delitem__("Ticket")
test_data.__delitem__("Age")
test_data.__delitem__("SibSp")
test_data.__delitem__("Parch")


pclass_converted = pd.get_dummies(test_data["Pclass"], prefix="is")
embarked_converted = pd.get_dummies(test_data["Embarked"], prefix="is")
sex_converted = pd.get_dummies(test_data["Sex"], prefix="is")

for pclass in pclass_converted:
    test_data[pclass] = pclass_converted[pclass]

for embark in embarked_converted:
    test_data[embark] = embarked_converted[embark]

for sex in sex_converted:
    test_data[sex] = sex_converted[sex]

# train_data.__delitem__("Pclass")
test_data.__delitem__("Embarked")
test_data.__delitem__("Sex")
# train_data = shuffle(train_data)

test_data = test_data.dropna()
labels = test_data["PassengerId"]
test_data.__delitem__("PassengerId")

test_data_norm = (test_data - test_data.mean()) / (test_data.max() - test_data.min())
np.asarray(test_data_norm, np.float64)

print(test_data_norm.head())
print(test_data_norm[1:12].head())

best_accuracy = 0
best_classifier = None
for i in range(0, 1000):
    train_data_input = train_data_norm[train_data_norm.columns[1:12]]
    train_data_label = train_data['Survived']

    data_train, data_test, labels_train, labels_test = train_test_split(train_data_input, train_data_label, test_size=0.20)
    classifier = neural_network.MLPClassifier(hidden_layer_sizes=[7, 2], max_iter=100000, learning_rate_init=0.03)
    classifier.fit(data_train, labels_train)

    predictions = classifier.predict(data_test)
    accuracy = accuracy_score(labels_test, predictions)
    if best_accuracy < accuracy:
        best_accuracy = accuracy
        best_classifier = classifier
        print('Iteration: '+str(i)+' '+str(best_accuracy))

predictions = best_classifier.predict(test_data_norm)
dictionary = {'PassengerId':labels,'Survived': predictions}
result = pd.DataFrame(data=dictionary)
print(result.head())
print(result.describe())

result[result.columns].to_csv(path_or_buf="results.csv", index=False)

