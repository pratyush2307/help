import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

d = {
    'miles_per_week': [37,39,46,51,88,17,18,20,21,22,23,24,25,27,28,29,30,31,32,33,34,38,40,42,
                       57,68,35,36,41,43,45,47,49,50,52,53,54,55,56,58,59,60,61,63,64,65,66,69,
                       70,72,73,75,76,77,78,80,81,82,83,84,85,86,87,89,91,92,93,95,96,97,98,99,
                       100,101,102,103,104,105,106,107,109,110,111,113,114,115,116,116,118,119,
                       120,121,123,124,126,62,67,74,79,90,112],
    'completed_50m_ultra': [
        'no','no','no','no','no','no','no','no','no','no','no','no','no','no','no','no','no','no','no','no','no',
        'no','no','no','no','no','yes','yes','yes','yes','no','yes','yes','yes','no','yes','yes','yes','yes','yes',
        'yes','yes','yes','no','yes','yes','yes','yes','yes','yes','yes','no','yes','yes','yes','yes','yes','yes',
        'yes','no','yes','yes','yes','yes','yes','yes','yes','no','yes','yes','yes','yes','yes','yes','yes','yes',
        'yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes',
        'yes','yes','yes','yes','yes'
    ]
}

df = pd.DataFrame(d)
enc = OrdinalEncoder(categories=[['yes', 'no']])
df['completed_50m_ultra'] = enc.fit_transform(df[['completed_50m_ultra']])

plt.scatter(df['miles_per_week'], df['completed_50m_ultra'])
plt.show()

x = df[['miles_per_week']]
y = df['completed_50m_ultra']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(model.score(x_test, y_test))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
