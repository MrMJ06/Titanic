import pandas as pd
from sklearn.utils import shuffle
from sklearn import neural_network
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

train_data = pd.read_csv("resources/train.csv")
train_data.__delitem__("Cabin")
train_data.__delitem__("PassengerId")
train_data.__delitem__("Name")
train_data.__delitem__("Ticket")

pclass_converted = pd.get_dummies(train_data["Pclass"], prefix="is")
embarked_converted = pd.get_dummies(train_data["Embarked"], prefix="is")
sex_converted = pd.get_dummies(train_data["Sex"], prefix="is")

for pclass in pclass_converted:
    train_data[pclass] = pclass_converted[pclass]

for embark in embarked_converted:
    train_data[embark] = embarked_converted[embark]

for sex in sex_converted:
    train_data[sex] = sex_converted[sex]

train_data.__delitem__("Pclass")
train_data.__delitem__("Embarked")
train_data.__delitem__("Sex")

train_data = train_data.dropna()
train_data = shuffle(train_data)

train_data_input = train_data[train_data.columns[1:13]]
train_data_label = train_data['Survived']

data_train, data_test, labels_train, labels_test = train_test_split(train_data_input, train_data_label, test_size=0.20, random_state=42)

classifier = neural_network.MLPClassifier(hidden_layer_sizes=[7], max_iter=100000, learning_rate_init=0.03)
classifier.fit(data_train, labels_train)

predictions = classifier.predict(data_test)
print(predictions)
print(accuracy_score(labels_test, predictions))
