# Figure 2.1: A Typical Root Finding Problem

import numpy as np
import matplotlib.pyplot as plt

def create_figure_2_1(save_figure=True):
    # Define the function and equation components
    x = np.linspace(-4, 4, 400)
    f1 = 3 * np.sin(x) + 9
    f2 = x**2 - np.cos(x)
    diff = f1 - f2

    # Find the intersection points numerically
    from scipy.optimize import fsolve

    def equation(x):
        return 3 * np.sin(x) + 9 - (x**2 - np.cos(x))

    # Initial guesses for solutions
    x_intersections = fsolve(equation, [-2, 3])
    y_intersections = 3 * np.sin(x_intersections) + 9

    # Create the plots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Left plot: Functions
    axs[0].plot(x, f1, 'k', label=r'$3\sin(x)+9$')
    axs[0].plot(x, f2, 'b--', label=r'$x^2 - \cos(x)$')
    axs[0].scatter(x_intersections, y_intersections, color='red', s=100, marker='*')  # Mark intersections
    axs[0].set_title(r'$3\sin(x)+9 = x^2 - \cos(x)$')
    axs[0].legend()
    axs[0].grid(True)

    # Right plot: Difference function
    axs[1].plot(x, diff, 'g-.', label=r'$(3\sin(x)+9)-(x^2-\cos(x))$')
    axs[1].scatter(x_intersections, np.zeros_like(x_intersections), color='red', s=100, marker='*')  # Mark zeros
    axs[1].set_title(r'$(3\sin(x)+9)-(x^2-\cos(x))=0$')
    axs[1].legend()
    axs[1].grid(True)

    if save_figure:
        fig.savefig("figures/Figure_2_1.png", dpi=300)

    plt.tight_layout()
    plt.show()

def create_figure_2_2(save_figure=True):

    # True root of x^2 - 2 = 0 (sqrt(2))
    true_root = np.sqrt(2)

    # Bisection method setup
    a, b = 0, 2  # Initial interval
    iterations = 30  # Number of iterations to display
    midpoints = []
    errors = []

    for _ in range(iterations):
        mid = (a + b) / 2
        midpoints.append(mid)
        errors.append(abs(mid - true_root))
        if (mid**2 - 2) * (a**2 - 2) < 0:
            b = mid
        else:
            a = mid

    # Plot error vs iteration
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(range(iterations), errors, color="blue", marker="*", label="Absolute Error")

    # Labels and title
    ax.set_xlabel("Iteration Number", fontsize=12)
    ax.set_ylabel("Absolute Error", fontsize=12)
    ax.set_title("Bisection Method Error vs Iteration", fontsize=14)
    ax.grid(True)

    if save_figure:
        fig.savefig("figures/Figure_2_2.png", dpi=300)

    # Display the plot
    plt.show()

def create_figure_2_3(save_figure=True):

    # Recreate the log2(error) vs iteration plot for the bisection method

    # True root of x^2 - 2 = 0 (sqrt(2))
    true_root = np.sqrt(2)

    # Bisection method setup
    a, b = 0, 2  # Initial interval
    iterations = 30  # Number of iterations to display
    midpoints = []
    errors = []

    for _ in range(iterations):
        mid = (a + b) / 2
        midpoints.append(mid)
        errors.append(abs(mid - true_root))
        if (mid**2 - 2) * (a**2 - 2) < 0:
            b = mid
        else:
            a = mid

    # Convert absolute error to log base 2 scale
    log_errors = np.log2(errors)

    # Fit a linear regression line for visualization
    coeffs = np.polyfit(range(iterations), log_errors, 1)  # Linear fit
    trendline = np.polyval(coeffs, range(iterations))

    # Plot log2(error) vs iteration
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(range(iterations), log_errors, color="blue", marker="*", label="Log2 Absolute Error")
    ax.plot(range(iterations), trendline, "r--", label="Linear Fit")

    # Labels and title
    ax.set_xlabel("Iteration Number", fontsize=12)
    ax.set_ylabel("Base 2 Log of Absolute Error", fontsize=12)
    ax.set_title("Bisection Method log2(Error) vs Iteration", fontsize=14)
    ax.legend()
    ax.grid(True)

    if save_figure:
        fig.savefig("figures/Figure_2_3.png", dpi=300)

    # Display the plot
    plt.show()

def create_figure_2_4(save_figure=True):

    # True root of x^2 - 2 = 0 (sqrt(2))
    true_root = np.sqrt(2)

    # Bisection method setup
    a, b = 0, 2  # Initial interval
    iterations = 30  # Number of iterations to display
    midpoints = []
    errors = []

    for _ in range(iterations):
        mid = (a + b) / 2
        midpoints.append(mid)
        errors.append(abs(mid - true_root))
        if (mid**2 - 2) * (a**2 - 2) < 0:
            b = mid
        else:
            a = mid

    # Convert absolute error to log base 2 scale
    log_errors = np.log2(errors)

    # Shifted data for iteration k vs k+1
    log_errors_k = log_errors[:-1]  # log2(error) at iteration k
    log_errors_k1 = log_errors[1:]  # log2(error) at iteration k+1

    # Fit a linear regression line for visualization
    coeffs = np.polyfit(log_errors_k, log_errors_k1, 1)  # Linear fit
    trendline = np.polyval(coeffs, log_errors_k)

    # Plot log2(error) at k+1 vs log2(error) at k
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(log_errors_k, log_errors_k1, color="blue", marker="*", label="Log2 Absolute Error")
    ax.plot(log_errors_k, trendline, "r--", label="Linear Fit")

    # Labels and title
    ax.set_xlabel("Base 2 Log of Error at Iteration k", fontsize=12)
    ax.set_ylabel("Base 2 Log of Error at Iteration k+1", fontsize=12)
    ax.set_title("Bisection Method log2(Error) vs log2(Error)", fontsize=14)
    ax.legend()
    ax.grid(True)

    if save_figure:
        fig.savefig("figures/Figure_2_4.png", dpi=300)

    # Display the plot
    plt.show()

def create_figure_2_5(save_figure=True):
    # True root of x^2 - 2 = 0 (sqrt(2))
    true_root = np.sqrt(2)

    # Bisection method setup
    a, b = 0, 2  # Initial interval
    iterations = 30  # Number of iterations to display
    midpoints = []
    errors = []

    for _ in range(iterations):
        mid = (a + b) / 2
        midpoints.append(mid)
        errors.append(abs(mid - true_root))
        if (mid**2 - 2) * (a**2 - 2) < 0:
            b = mid
        else:
            a = mid

    # Convert absolute error to log base 10 scale
    log_errors = np.log10(errors)

    # Shifted data for iteration k vs k+1
    log_errors_k = log_errors[:-1]  # log10(error) at iteration k
    log_errors_k1 = log_errors[1:]  # log10(error) at iteration k+1

    # Fit a linear regression line for visualization
    coeffs = np.polyfit(log_errors_k, log_errors_k1, 1)  # Linear fit
    trendline = np.polyval(coeffs, log_errors_k)

    # Plot log10(error) at k+1 vs log10(error) at k
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(log_errors_k, log_errors_k1, color="blue", marker="*", label="Log10 Absolute Error")
    ax.plot(log_errors_k, trendline, "r--", label="Linear Fit")

    # Labels and title
    ax.set_xlabel("Base 10 Log of Error at Iteration k", fontsize=12)
    ax.set_ylabel("Base 10 Log of Error at Iteration k+1", fontsize=12)
    ax.set_title("Bisection Method log10(Error) vs log10(Error)", fontsize=14)
    ax.legend()
    ax.grid(True)

    if save_figure:
        fig.savefig("figures/Figure_2_5.png", dpi=300)

    # Display the plot
    plt.show()

def create_figure_2_6(save_figure=True):

    # Create the figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    # Define the function
    f = lambda x: x**2 - 2

    # Generate x values for the function plot
    x_vals = np.linspace(-0.1, 2.1, 400)
    y_vals = f(x_vals)

    # True root of x^2 - 2 = 0 (sqrt(2))
    true_root = np.sqrt(2)

    # Initial search intervals for Bisection Method visualization
    intervals = [
        (0, 2),   # First subplot interval
        (1, 2),   # Second subplot interval
        (0, 1.5)  # Third subplot interval
    ]

    # Titles for the subplots
    titles = [
        "Initial Search Interval = [0,2]",
        "Initial Search Interval = [1,2]",
        "Initial Search Interval = [0,1.5]"
    ]   

    # Plot each interval
    for ax, (a, b), title in zip(axes, intervals, titles):
        ax.plot(x_vals, y_vals, color='b', label=r"$f(x) = x^2 - 2$")
        ax.axhline(0, color='black', linewidth=1)  # x-axis

        # Scatter red dots at (a,0) and (b,0) on x-axis
        ax.scatter([a, b], [0, 0], color='red', s=100, marker="o", label="Initial Interval")

        # Scatter green star at true root
        ax.scatter(true_root, 0, color='green', s=120, marker="*", label="True Root")

        # Labels and title
        ax.set_xlabel(r"$x$", fontsize=12)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-0.1, 2.1)  # Extended x-axis limits
        ax.set_ylim(-2, 2)
        ax.grid(True)

    # Set common y-label
    axes[0].set_ylabel(r"$f(x)$", fontsize=12)

    if save_figure:
        fig.savefig("figures/Figure_2_6.png", dpi=300)

    # Adjust layout and show the plot
    fig.tight_layout()
    plt.show()

def create_figure_2_7(save_figure=True):

    # Create the figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)

    # Define the function
    f = lambda x: x**2 - 2

    # Generate x values for the function plot
    x_vals = np.linspace(-0.1, 2.1, 400)
    y_vals = f(x_vals)

    # True root of x^2 - 2 = 0 (sqrt(2))
    true_root = np.sqrt(2)

    # Initial search intervals for Bisection vs Regula Falsi visualization
    intervals = [
        (0, 2),   # First subplot interval
        (1, 2),   # Second subplot interval
        (0, 1.5)  # Third subplot interval
    ]

    # Titles for the subplots
    titles = [
        "Initial Search Interval = [0,2]",
        "Initial Search Interval = [1,2]",
        "Initial Search Interval = [0,1.5]"
    ]

    # Plot each interval with additional elements
    for ax, (a, b), title in zip(axes, intervals, titles):
        ax.plot(x_vals, y_vals, color='b', label=r"$f(x) = x^2 - 2$")  # Function curve
        ax.axhline(0, color='black', linewidth=1)  # x-axis

        # Scatter red dots at (a,0) and (b,0) on x-axis
        ax.scatter([a, b], [0, 0], color='red', s=100, marker="o", label="Initial Interval")

        # Scatter green star at true root
        ax.scatter(true_root, 0, color='green', s=120, marker="*", label="True Root")

        # Compute the secant line
        fa, fb = f(a), f(b)
        secant_slope = (fb - fa) / (b - a)
        secant_x_intercept = a - fa / secant_slope  # Solve for x when y = 0

        # Scatter blue dots at secant endpoints
        ax.scatter([a, b], [fa, fb], color='blue', s=100, marker="o", label="Secant Endpoints")

        # Plot the secant line
        ax.plot([a, b], [fa, fb], 'k--', linewidth=1)

        # Scatter a black x at the x-intercept of the secant line
        ax.scatter(secant_x_intercept, 0, color='black', s=100, marker="x", label="Secant X-Intercept")

        # Labels and title
        ax.set_xlabel(r"$x$", fontsize=12)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-0.1, 2.1)
        ax.set_ylim(-2.1, 2.1)
        ax.grid(True)

    # Set common y-label
    axes[0].set_ylabel(r"$f(x)$", fontsize=12)

    if save_figure:
        fig.savefig("figures/Figure_2_7.png", dpi=300)

    # Adjust layout and show the plot
    fig.tight_layout()
    plt.show()

