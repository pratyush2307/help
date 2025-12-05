import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

plt.scatter(df["sepal length (cm)"], df["sepal width (cm)"], c=df["target"])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()

X = df[iris.feature_names]
y = df["target"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = GaussianNB()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
