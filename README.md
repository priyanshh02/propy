import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
data = {
    'Month': [1, 2, 3, 4, 5, 6], 
    'Expense': [220, 240, 260, 280, 300, 320]  
}

df = pd.DataFrame(data)
X = df[['Month']] 
y = df['Expense']   

model = LinearRegression()
model.fit(X, y)

future_months = np.array([[7], [8], [9]]) 
predicted_expenses = model.predict(future_months)

for i, month in enumerate(future_months):
    print(f"Predicted expense for month {month[0]}: ${predicted_expenses[i]:.2f}")

plt.scatter(X, y, color='blue', label='Past Expenses')
plt.plot(X, model.predict(X), color='green', label='Trend Line')
plt.scatter(future_months, predicted_expenses, color='red', label='Predicted')
plt.xlabel('Month')
plt.ylabel('Expense ($)')
plt.title('Monthly Expense Prediction')
plt.legend()
plt.show()
