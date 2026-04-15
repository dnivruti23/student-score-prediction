import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample data
data = pd.DataFrame({
    'hours': [1, 2, 3, 4, 5, 6, 7, 8],
    'score': [10, 20, 30, 40, 50, 60, 70, 80]
})

X = data[['hours']]
y = data['score']

model = LinearRegression()
model.fit(X, y)

prediction = model.predict([[6]])
print("Predicted score:", prediction[0])