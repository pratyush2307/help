import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv("quikr_car.csv")

df = df[df["year"].str.isnumeric()]
df["year"] = df["year"].astype(int)

df = df[df["Price"] != "Ask For Price"]
df["Price"] = df["Price"].str.replace(",", "").astype(int)

df["kms_driven"] = df["kms_driven"].str.split().str[0].str.replace(",", "")
df = df[df["kms_driven"].str.isnumeric()]
df["kms_driven"] = df["kms_driven"].astype(int)

df = df[df["Price"] < 6000000]

X = df[["year", "kms_driven"]]
y = df["Price"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Price")
plt.show()
