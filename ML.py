import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Read the CSV file
df = None
try:
    df = pd.read_csv("AreaVsPrice.csv")
    print(df.head())  # Display the first few rows of the DataFrame
except Exception as e:
    print("Error reading the file:", e)

# If DataFrame was successfully read, proceed with plotting and testing
if df is not None:
    # Prepare data for linear regression
    X = df["Area"].values.reshape(-1, 1)  # Independent variable (Area)
    y = df["Price"].values  # Dependent variable (Price)

    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Generate predictions for the existing data
    y_pred = model.predict(X)

    # Plot scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df["Area"], df["Price"], color="blue", label="Actual Data", alpha=0.7)

    # Plot linear regression line
    plt.plot(df["Area"], y_pred, color="red", label="Linear Fit")

    # Add labels, title, and legend
    plt.title("Area vs Price")
    plt.xlabel("Area (sq. ft.)")
    plt.ylabel("Price (in $)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Test the model with new values
    test_areas = np.array([500, 1200, 2500, 5000]).reshape(-1, 1)  # Example areas
    predicted_prices = model.predict(test_areas)

    # Print the results
    for area, price in zip(test_areas.flatten(), predicted_prices):
        print(f"Predicted price for area {area} sq. ft. is ${price:.2f}")
