import numpy as np
import matplotlib.pyplot as plt

def estimate_coefficients(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    cross_deviation = np.sum((x - x_mean) * (y - y_mean))
    x_deviation = np.sum((x - x_mean) ** 2)

    slope = cross_deviation / x_deviation    
    intercept = y_mean - slope * x_mean

    return slope, intercept

def plot_regression_line(x, y, slope, intercept):
    plt.scatter(x, y, color="m", marker="o", s=30, label="Data points")

    y_pred = slope * x + intercept  # Generate predictions
    plt.plot(x, y_pred, color="g", label="Regression line")
    plt.title('Linear Regression')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(color='gray', linestyle='--', linewidth=0.5)

    plt.show()

def main():
    np.random.seed(42)  # Set seed for reproducibility
    X = np.random.uniform(1, 20, 100)  # Generate random x values
    
    slope_true = 1.5  # True slope of the line
    intercept_true = 3  # True intercept
    noise = np.random.normal(0, 1, 100)  # Adding some noise to the data
    
    # Generate y values based on a linear relation with x and some random noise
    y = intercept_true + slope_true * X + noise

    # Estimate the regression coefficients
    slope, intercept = estimate_coefficients(X, y)
    print(f"Estimated regression coefficients:\nSlope = {slope:.2f}, Intercept = {intercept:.2f}")

    # Plot the regression line
    plot_regression_line(X, y, slope, intercept)

if __name__ == "__main__":
    main()
