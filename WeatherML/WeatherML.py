
import pandas as pd #Pandas is for Data processing. 
import matplotlib.pyplot as plt #plotting and visually understanding the data.
from sklearn.model_selection import train_test_split #split the data into testing and training set
from sklearn.preprocessing import LabelEncoder #converts things like "monsoon", "summer" into numeric labels like 0 1 2
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC # SUPPORT VECTOR CLASSIFIER
from sklearn.metrics import classification_report #Performance evaluation
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv("dataset.csv")
display(df)


encoder = LabelEncoder()
df['Season'] = encoder.fit_transform(df['Season'])

display(df)

scaler = MinMaxScaler()

# Apply the scaler to the relevant columns
df[['Temperature', 'Humidity', 'Windspeed']] = scaler.fit_transform(df[['Temperature', 'Humidity', 'Windspeed']])

# Check the normalized data
print(df[['Temperature', 'Humidity', 'Windspeed']].head())

df.to_csv("datasetn.csv")

import seaborn as sns
sns.pairplot(df, hue='Rain', vars=['Temperature', 'Humidity', 'Windspeed'])
plt.show()

X = df[['Temperature', 'Humidity', 'Windspeed']]
Y = df['Rain']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#test_size is for splitting. basically 0.2 means 20% of data will be used for testing.


#  NOW LETS TRAIN THE MODEL

model = SVC()
model.fit(X_train,Y_train)

predict = model.predict(X_test)
print(X_test)
print(predict)

temp = float(input("Enter the temperature: "))
hum = float(input("Humidity? "))
wind = float(input("Wind "))

user = [(temp, hum, wind)]
print(model.predict(user))

print(classification_report(Y_test, model.predict(X_test)))


