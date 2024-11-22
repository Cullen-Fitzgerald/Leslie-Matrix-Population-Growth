import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.optimize import fsolve
from scipy.integrate import quad
from scipy.optimize import minimize_scalar


def polynomial_solver():
    def denominator(lam):
        return lam**10 - 2.025 * lam**7 - 0.648 * lam**6 + 1.366875 * lam**4 + \
            0.8748 * lam**3 - 0.307546875 * lam - 0.295245

    def equation(lam):
        return (
            (lam**16) / denominator(lam) -
            (2.7 * lam**13) / denominator(lam) -
            (1.296 * lam**12) / denominator(lam) -
            (0.245025 * lam**11) / denominator(lam) +
            (2.70702 * lam**10) / denominator(lam) +
            (2.6244 * lam**9) / denominator(lam) +
            (0.916079625 * lam**8) / denominator(lam) -
            (1.01728305 * lam**7) / denominator(lam) -
            (1.75414896 * lam**6) / denominator(lam) -
            (0.901788946875 * lam**5) / denominator(lam) -
            (0.0432902981250001 * lam**4) / denominator(lam) +
            (0.375197346 * lam**3) / denominator(lam) +
            (0.266675433046875 * lam**2) / denominator(lam) +
            (0.08056313409375 * lam) / denominator(lam) +
            0.00789189885 / denominator(lam)
        )

    initial_guess = 1.0

    solution = fsolve(equation, initial_guess)

    print(f"The solution for lambda is: {solution[0]}")

def smaller_polynomial_solver():

    def equation(lam):
        return (
            (0.0675 / (lam**3)) +
            (0.0648 / (lam**4)) +
            (0.0245025 / (lam**5))
            - 1
        )

    initial_guess = 1.0

    solution = fsolve(equation, initial_guess)

    print(f"The solution for lambda is: {solution[0]}")

    print(equation(solution[0]) == 0)

import numpy as np
import matplotlib.pyplot as plt


def culled_ages(years):
    population = np.array([900, 1050, 1000, 725, 382.5, 200])
    breeding_rate = np.array([0, 0, 1.5, 1.6, 1.1, 0.4])
    survival_rate = np.array([0.6, 0.75, 0.9, 0.55, 0.3, 0])
    size = len(population)
    leslie_matrix = np.zeros((size, size))
    leslie_matrix[0] = breeding_rate
    for i in range(1, size):
        leslie_matrix[i, i - 1] = survival_rate[i - 1]
    eigenvalues, eigenvectors = np.linalg.eig(leslie_matrix)
    growth_rate = np.max(np.real(eigenvalues))
    stable_distribution = np.real(eigenvectors[:, np.argmax(np.real(eigenvalues))])
    stable_distribution /= np.sum(stable_distribution)
    culling_factor = 1 - (1 / growth_rate)
    kept_factor = 1 / growth_rate
    populations = [population.copy()]
    for year in range(years):
        next_population = np.dot(leslie_matrix, populations[-1])
        next_population *= kept_factor
        populations.append(next_population.copy())
    populations = np.array(populations)
    plt.figure(figsize=(10, 6))
    for age_group in range(size):
        plt.plot(range(years + 1), populations[:, age_group], marker='o', linestyle='-', label=f'Age Group {age_group} - {age_group + 1}')
    plt.title(f'Population for Each Age Group Over {years} Years')
    plt.xlabel('Years')
    plt.ylabel('Population Size')
    plt.legend()
    plt.grid(True)
    plt.show()
    final_population = populations[-1]
    total = final_population.sum()
    distribution = (final_population / total) * 100
    print(f"The Stabilised Distribution is: {distribution}")

import numpy as np
import matplotlib.pyplot as plt

def farmed_ages(years, population_limit):
    population = np.array([900, 1050, 1000, 725, 382.5, 200])
    breeding_rate = np.array([0, 0, 0.15, 0.16, 0.11, 0.04])
    survival_rate = np.array([0.6, 0.75, 0.9, 0.55, 0.3, 0])
    size = len(population)
    leslie_matrix = np.zeros((size, size))
    leslie_matrix[0] = breeding_rate
    for i in range(1, size):
        leslie_matrix[i, i - 1] = survival_rate[i - 1]
    eigenvalues, eigenvectors = np.linalg.eig(leslie_matrix)
    growth_rate = np.max(np.real(eigenvalues))
    stable_distribution = np.real(eigenvectors[:, np.argmax(np.real(eigenvalues))])
    stable_distribution /= np.sum(stable_distribution)
    kept_factor = 1 / growth_rate
    populations = [population.copy()]
    for year in range(years):
        next_population = np.dot(leslie_matrix, populations[-1])
        if next_population.sum() < population_limit:
            next_population *= kept_factor
        populations.append(next_population.copy())
    populations = np.array(populations)
    plt.figure(figsize=(10, 6))
    for age_group in range(size):
        plt.plot(range(years + 1), populations[:, age_group], marker='o', linestyle='-', label=f'Age Group {age_group } - {age_group + 1}')
    plt.title(f'Population for Each Age Group Over {years} Years')
    plt.xlabel('Years')
    plt.ylabel('Population Size')
    plt.legend()
    plt.grid(True)
    plt.show()
    final_population = populations[-1]
    total = final_population.sum()
    distribution = (final_population / total) * 100
    print(f"The Stabilised Distribution is: {distribution}")

def culled(years):
    population = np.array([900, 1050, 1000, 725, 382.5, 200])
    breeding_rate = np.array([0, 0, 1.5, 1.6, 1.1, 0.4])
    survival_rate = np.array([0.6, 0.75, 0.9, 0.55, 0.3, 0])
    size = len(population)
    leslie_matrix = np.zeros((size, size))
    leslie_matrix[0] = breeding_rate
    for i in range(1, size):
        leslie_matrix[i, i - 1] = survival_rate[i - 1]
    print("Leslie Matrix:")
    print(leslie_matrix)
    eigenvalues, eigenvectors = np.linalg.eig(leslie_matrix)
    print(np.real(eigenvectors))
    print(f"The eigenvalues are {eigenvalues}")
    growth_rate = np.max(np.real(eigenvalues))
    stable_distribution = np.real(eigenvectors[:, np.argmax(np.real(eigenvalues))])
    stable_distribution = stable_distribution / np.sum(stable_distribution)
    print(f"\nGrowth Rate (Dominant Eigenvalue): {growth_rate}")
    print(f"Stable Age Distribution: {stable_distribution}")
    culling_factor = 1 - (1 / growth_rate)
    kept_factor = 1 / growth_rate
    print(f"Culling Factor to Stabilize Population: {culling_factor}")
    populations = [population.copy()]  
    for year in range(years):
        next_population = np.dot(leslie_matrix, populations[-1])
        next_population *= kept_factor
        populations.append(next_population.copy())  
        print(sum(next_population), year)
    populations = np.array(populations)
    print(populations, year)
    total_population = populations.sum(axis=1)
    plt.figure(figsize=(10, 6))
    plt.plot(range(years + 1), total_population, marker='o', linestyle='-', color='b', label='Culled Population')
    plt.title(f'Total Population Over {years} Years')
    plt.xlabel('Years')
    plt.ylabel('Total Population')
    plt.legend()
    plt.grid(True)
    plt.show()

def dominant_eigenvector():
    # Define the matrix A
    A = np.array([
        [0, 0, 0.15, 0.16, 0.11, 0.04],
        [0.6, 0, 0, 0, 0, 0],
        [0, 0.75, 0, 0, 0, 0],
        [0, 0, 0.9, 0, 0, 0],
        [0, 0, 0, 0.55, 0, 0],
        [0, 0, 0, 0, 0.3, 0]
    ])

    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # Find the index of the largest eigenvalue (in magnitude)
    dominant_index = np.argmax(np.abs(eigenvalues))

    # Extract the dominant eigenvector
    dominant_eigenvector = eigenvectors[:, dominant_index]

    # Normalize the dominant eigenvector (optional, for readability)
    dominant_eigenvector = dominant_eigenvector / np.linalg.norm(dominant_eigenvector)

    # Print the results
    print("Dominant Eigenvalue:", eigenvalues[dominant_index])
    print("Dominant Eigenvector:", dominant_eigenvector)


def farming(years, population_limit):
    population = np.array([900, 1050, 1000, 725, 382.5, 200])
    breeding_rate = np.array([0, 0, 0.15, 0.16, 0.11, 0.04])
    survival_rate = np.array([0.6, 0.75, 0.9, 0.55, 0.3, 0])
    size = len(population)
    leslie_matrix = np.zeros((size, size))
    leslie_matrix[0] = breeding_rate
    for i in range(1, size):
        leslie_matrix[i, i - 1] = survival_rate[i - 1]
    print("Leslie Matrix:")
    print(leslie_matrix)
    eigenvalues, eigenvectors = np.linalg.eig(leslie_matrix)
    print(np.real(eigenvectors))
    print(f"The eigenvalues are {eigenvalues}")
    growth_rate = np.max(np.real(eigenvalues))
    stable_distribution = np.real(eigenvectors[:, np.argmax(np.real(eigenvalues))])
    stable_distribution = stable_distribution / np.sum(stable_distribution)
    print(f"\nGrowth Rate (Dominant Eigenvalue): {growth_rate}")
    print(f"Stable Age Distribution: {stable_distribution}")
    kept_factor = 1 / growth_rate  
    print(f"Culling Factor to Stabilize Population: {1 - kept_factor}")
    populations = [population.copy()]
    for year in range(years):
        next_population = np.dot(leslie_matrix, populations[-1])
        if next_population.sum() < population_limit:
            next_population *= kept_factor
        populations.append(next_population.copy())
        print(next_population, f"It is year {year}")
    populations = np.array(populations)
    total_population = populations.sum(axis=1)
    plt.figure(figsize=(10, 6))
    plt.plot(range(years + 1), total_population, marker='o', linestyle='-', color='b', label='Farmed Population')
    plt.title(f'Total Population Dynamics Over {years} Years')
    plt.xlabel('Years')
    plt.ylabel('Total Population')
    plt.legend()
    plt.grid(True)
    plt.show()

def projectile_motion():

    # Constants
    v0 = 25  # Initial velocity in m/s
    g = 9.81  # Acceleration due to gravity in m/s^2

    # Function to calculate the arc length for a given angle
    def arc_length(theta):
        # Define the time of flight
        T = (2 * v0 * np.sin(theta)) / g
        
        # Define the integrand for the arc length
        def integrand(t):
            dx_dt = v0 * np.cos(theta)
            dy_dt = v0 * np.sin(theta) - g * t
            return np.sqrt(dx_dt**2 + dy_dt**2)
        
        # Calculate the arc length using numerical integration
        length, _ = quad(integrand, 0, T)
        return length

    # Optimize to find the angle that maximizes the arc length
    result = minimize_scalar(lambda theta: -arc_length(theta), bounds=(0, np.pi/2), method='bounded')

    # Optimal angle in degrees
    optimal_angle_degrees = np.degrees(result.x)
    optimal_arc_length = arc_length(result.x)

    # Output results
    print(f"Optimal Angle: {optimal_angle_degrees:.2f} degrees")
    print(f"Maximum Arc Length: {optimal_arc_length:.2f} meters")


def unaltered_population(years):
    population = np.array([900, 1050, 1000, 725, 382.5, 200])
    breeding_rate = np.array([0, 0, 0.15, 0.16, 0.11, 0.04])
    survival_rate = np.array([0.6, 0.75, 0.9, 0.55, 0.3, 0])

    size = len(population)
    leslie_matrix = np.zeros((size, size))

    leslie_matrix[0] = breeding_rate

    for i in range(1, size):
        leslie_matrix[i, i - 1] = survival_rate[i - 1]

    print("Leslie Matrix:")
    print(leslie_matrix)

    unaltered_populations = [population.copy()]

    for year in range(years):
        next_unaltered_population = np.dot(leslie_matrix, unaltered_populations[-1])
        unaltered_populations.append(next_unaltered_population.copy())

    unaltered_populations = np.array(unaltered_populations)
    total_unaltered_population = unaltered_populations.sum(axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(range(years + 1), total_unaltered_population, 
             marker='o', linestyle='dashed', color='r', label='Unaltered Population')

    plt.title(f'Total Population Dynamics Over {years} Years')
    plt.xlabel('Years')
    plt.ylabel('Total Population')
    plt.legend()
    plt.grid(True)
    plt.show()

projectile_motion()