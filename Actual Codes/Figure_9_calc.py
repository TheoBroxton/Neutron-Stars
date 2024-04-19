# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 12:28:44 2024

@author: Joshua Edwards
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, hbar, G, m_e, m_p, m_n, pi

# Constants and parameters

M0 = 1.9891e30  # Solar mass in kg
R0 = G * M0 / c**2  # Half Schwarzchild radius in km
m_N = 0.5 * (m_p + m_n)
eta_ns = 1  # Ratio of A/Z for neutron star

# Equation of state constant
epsilon0 = (m_n**4 * c**5) / (np.pi**2 * hbar**3)

# Function to find the root using Newton-Raphson method
def find_root(func, x_guess):
    tol = 1e-6
    max_iter = 1000
    for _ in range(max_iter):
        f_x = func(x_guess)
        f_prime_x = (func(x_guess + tol) - func(x_guess)) / tol
        x_guess -= f_x / f_prime_x
        if abs(f_x) < tol:
            break
    return x_guess

# Define the function representing p(x) - p
def func_p_minus_p(x, p):
    p_func = (epsilon0/24) * ((2 * x**3 - 3 * x) * (1 + x**2)**0.5 + 3 * np.arcsinh(x))
    return p_func - p

# Define the function representing epsilon(x)
def epsilon_x(x):
    return (epsilon0/8) * ((2 * x**3 + x) * (1 + x**2)**0.5 - np.arcsinh(x))

def grad(time, state):
    """
    Compute the gradients of the state variables.

    Parameters:
        time (float): Current time.
        state (array-like): Array containing the current values of the state variables.

    Returns:
        array-like: Gradients of the state variables.
    """
    r, p, mbar = state

    if p <= 0:
        return np.array([0, 0, 0])

    # Use Newton-Raphson method to find the root of p(x) - p = 0
    x_guess = ((p * 15) / epsilon0)**(1/5)
    x_root = find_root(lambda x: func_p_minus_p(x, p), x_guess)
    #print("x root: ", x_root, "pressure: ", p)
    # Compute epsilon using the given equation
    epsilon = epsilon_x(x_root)
    #print(epsilon)
    m = mbar * M0

    # Compute dp/dr and dmbar/dr
    dp_dr = -(((G * epsilon * m) / (c**2 * r**2)) + ((4 * pi * r * p * epsilon * G) / (c**4))) * (1 +(p/epsilon)) * (1 - ((2 * G * m) / (c**2 * r)))**-1
    
    dmbar_dr = (4 * pi * (r**2) * epsilon) / ((c**2) * M0)

    return np.array([1, dp_dr, dmbar_dr])

def rk4_step(grad, time, state, step_size):
    """
    Perform a single step of the fourth-order Runge-Kutta method.

    Parameters:
        grad (function): Function that computes the gradients of the state variables.
        time (float): Current time.
        state (array-like): Array containing the current values of the state variables.
        step_size (float): Step size for integration.

    Returns:
        tuple: Tuple containing the new time and state after the RK4 step.
    """
    k1 = grad(time, state) * step_size
    k2 = grad(time + step_size / 2, state + k1 / 2) * step_size
    k3 = grad(time + step_size / 2, state + k2 / 2) * step_size
    k4 = grad(time + step_size, state + k3) * step_size

    new_time = time + step_size
    new_state = state + (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6)
    return new_time, new_state

def ns_calculation():
    # Generate logarithmically spaced initial pressures
    initial_pressures_ns = np.logspace(np.log10(1e29), np.log10(2e38), 150)

    # Initialize lists to store total masses and total radii for each pressure
    total_masses_ns = []
    total_radii_ns = []

    # Initialize variables to store the maximum total mass and its corresponding pressure and radius
    max_total_mass = 0
    max_pressure = 0
    corresponding_radius = 0

    # Loop through each initial pressure
    for p0 in initial_pressures_ns:
        # Define initial state vector
        state0 = np.array([1e-10, p0, 0.0])  # r=1e-10 to avoid division by zero

        # Specify step size and final time
        step_size = 1e2  # Adjust as needed
        final_time = 1e7  # Adjust as needed

        # Perform RK4 integration
        times = np.arange(0, final_time, step_size)
        state = state0
        for t in times:
            new_time, new_state = rk4_step(grad, t, state, step_size)
            state = new_state

            # Check for non-positive pressure
            if new_state[1] <= 0:
                break

        # Append total mass and total radius to lists
        total_masses_ns.append(new_state[2])
        total_radii_ns.append(new_state[0] / 1000)

        # Update maximum total mass and its corresponding pressure and radius
        if new_state[2] > max_total_mass:
            max_total_mass = new_state[2]
            max_pressure = p0
            corresponding_radius = new_state[0]

    # Print the maximum total mass and its corresponding pressure and radius
    print("Maximum Total Mass:", max_total_mass)
    print("Corresponding Initial Pressure:", max_pressure)
    print("Corresponding Total Radius:", corresponding_radius)

    # Plot total mass and total radius vs. initial pressure
    plot_mass_radius_vs_pressure(initial_pressures_ns, total_masses_ns,
                                 total_radii_ns, 'NS_varying_p0', 'log')

def plot_mass_radius_vs_pressure(pressures, total_masses, total_radii,
                                 filename='varying_p0', scale='linear'):
    """
    Plot the mass and radius of neutron stars against the initial pressures.

    Parameters:
        pressures (array-like): Array of initial pressures.
        total_masses (array-like): Array of total masses corresponding to each
                                   initial pressure.
        total_radii (array-like): Array of total radii corresponding to each
                                  initial pressure.
        filename (str): Name of the file to save the plot. Defaults to saving
                        in a Figures subdirectory.
        scale(str): Type of scale for plotting.
    """
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_ylabel('Total Radius (km)', color=color)
    ax1.set_xlabel('Initial Pressure (Pa)')
    ax1.plot(pressures, total_radii, color=color, linestyle = '--', label='Total Radius')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.margins(x=0)
    ax1.set_xscale(scale)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Total Mass ($M_0$)', color=color)
    ax2.plot(pressures, total_masses, color=color, label='Total Mass')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.margins(x=0)
    ax2.set_xscale(scale)

    fig.tight_layout()
    fig.legend(loc="upper right")

    plt.show()

if __name__ == "__main__":
    ns_calculation()
