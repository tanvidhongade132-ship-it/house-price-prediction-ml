import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Create a sample housing dataset
data = {
    "Area": [1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800],
    "Bedrooms": [2, 2, 3, 3, 3, 4, 4, 5],
    "Bathrooms": [1, 2, 2, 2, 3, 3, 3, 4],
    "Price": [2500000, 3000000, 4000000, 4500000,
              5500000, 6000000, 7000000, 8000000]
}

df = pd.DataFrame(data)

# Select input features and target
X = df[["Area", "Bedrooms", "Bathrooms"]]
y = df["Price"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)

print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

# Predict the price of a new house
new_house = pd.DataFrame({
    "Area": [1600],
    "Bedrooms": [3],
    "Bathrooms": [2]
})

predicted_price = model.predict(new_house)

print("Predicted House Price: ₹", round(predicted_price[0], 2))
