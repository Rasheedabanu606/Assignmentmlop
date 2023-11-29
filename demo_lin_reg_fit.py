import pickle
from sklearn.linear_model import LinearRegression

Experience = [[1], [2], [3], [4], [5]]
Salary = [20000, 30000, 50000, 80000, 100000]

model = LinearRegression()
model.fit(Experience, Salary)

# Save the trained model using pickle
with open('modelsal.pkl', 'wb') as file:
    pickle.dump(model, file)
