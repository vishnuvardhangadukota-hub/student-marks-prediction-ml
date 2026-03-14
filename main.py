import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = {
    "Hours":[1,2,3,4,5,6,7,8],
    "Marks":[20,30,40,50,60,70,80,90]
}

df = pd.DataFrame(data)

X = df[["Hours"]]
y = df["Marks"]

model = LinearRegression()
model.fit(X, y)

# user input
hours = float(input("Enter study hours: "))

prediction = model.predict([[hours]])

print("Predicted marks:", prediction[0])

plt.scatter(df["Hours"], df["Marks"])
plt.plot(df["Hours"], model.predict(X))

plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks Prediction")

plt.show()