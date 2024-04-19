# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 13:15:35 2024

@author: Joshua Edwards
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, hbar, G, m_e, m_p, m_n, pi

# Constants and parameters
gamma_nrel = 5/3
M0 = 1.9891e30  # Solar mass in kg
R0 = G * M0 / c**2  # Half Schwarzchild radius in km
m_N = 0.5 * (m_p + m_n)
eta_ns = 1  # Ratio of A/Z for neutron star

# Equation of state constant
K_nrel = ((hbar**2) / (15 * pi**2 * m_n)) * \
    ((3 * pi**2) / (m_n * eta_ns * c**2))**(gamma_nrel)


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


def grad_newtonian(time, state):
    """
    Compute the gradients of the state variables for the Newtonian calculation.

    Parameters:
        time (float): Current time.
        state (array-like): Array containing the current values of the state variables.

    Returns:
        array-like: Gradients of the state variables.
    """
    r, p, mbar = state

    if p <= 0:
        return np.array([0, 0, 0])

    # Compute dp/dr and dmbar/dr
    dp_dr = -(R0 * p**(1/gamma_nrel) * mbar) / \
        (r**2 * K_nrel**(1/gamma_nrel))
    dmbar_dr = (4 * pi * r**2 * p**(1/gamma_nrel)) / \
        (M0 * c**2 * K_nrel**(1/gamma_nrel))

    return np.array([1, dp_dr, dmbar_dr])


def grad_TOV(time, state):
    """
    Compute the gradients of the state variables for the TOV calculation.

    Parameters:
        time (float): Current time.
        state (array-like): Array containing the current values of the state variables.

    Returns:
        array-like: Gradients of the state variables.
    """
    r, p, mbar = state

    if p <= 0:
        return np.array([0, 0, 0])

    epsilon = (p/K_nrel)**(1/gamma_nrel)
    m = mbar * M0

    # Compute dp/dr and dmbar/dr
    dp_dr = -(((G * epsilon * m) / (c**2 * r**2)) + ((4 * pi * r * p * epsilon * G) / (c**4))) * (1 +(p/epsilon)) * (1 - ((2 * G * m) / (c**2 * r)))**-1
    dmbar_dr = (4 * pi * r**2 * p**(1/gamma_nrel)) / \
        (M0 * c**2 * K_nrel**(1/gamma_nrel))

    return np.array([1, dp_dr, dmbar_dr])


def ns_calculation(grad_function):
    # Generate logarithmically spaced initial pressures
    initial_pressures_ns = np.logspace(np.log10(2.5e30), np.log10(4e32), 100)

    # Initialize lists to store total masses and total radii for each pressure
    total_masses_ns = []
    total_radii_ns = []

    # Loop through each initial pressure
    for p0 in initial_pressures_ns:
        # Define initial state vector
        state0 = np.array([1e-10, p0, 0.0])  # r=1e-10 to avoid division by zero

        # Specify step size and final time
        step_size = 8e1  # Adjust as needed
        final_time = 1e7  # Adjust as needed

        # Perform RK4 integration
        times = np.arange(0, final_time, step_size)
        state = state0
        for t in times:
            new_time, new_state = rk4_step(grad_function, t, state, step_size)
            state = new_state

            # Check for non-positive pressure
            if new_state[1] <= 0:
                break

        # Append total mass and total radius to lists
        total_masses_ns.append(new_state[2])
        total_radii_ns.append(new_state[0] / 1000)

    return initial_pressures_ns, total_masses_ns, total_radii_ns


def plot_mass_radius_vs_pressure(pressures, total_masses, total_radii, labels, line_thicknesses, scale='linear'):
    """
    Plot the mass and radius of neutron stars against the initial pressures.

    Parameters:
        pressures (list of array-like): Arrays of initial pressures.
        total_masses (list of array-like): Arrays of total masses corresponding to each
                                   initial pressure.
        total_radii (list of array-like): Arrays of total radii corresponding to each
                                  initial pressure.
        labels (list of str): Labels for each set of data.
        line_thicknesses (list of float): Thickness of lines for each set of data.
        scale(str): Type of scale for plotting.
    """
    fig, ax1 = plt.subplots()

    colors = ['tab:red', 'tab:blue']
    ax1.set_xlabel('Initial Pressure (Pa)')
    ax1.set_ylabel('Total Radius (km)', color=colors[0])

    for i in range(len(pressures)):
        ax1.plot(pressures[i], total_radii[i], color=colors[i], linewidth=line_thicknesses[i], label=f'Total Radius ({labels[i]})')
    ax1.tick_params(axis='y', labelcolor=colors[0])
    ax1.set_xscale(scale)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.set_ylabel('Total Mass ($M_0$)', color=colors[1])
    for i in range(len(pressures)):
        ax2.plot(pressures[i], total_masses[i], color=colors[i], linestyle='--', linewidth=line_thicknesses[i], label=f'Total Mass ({labels[i]})')
    ax2.tick_params(axis='y', labelcolor=colors[1])
    ax2.legend(loc="upper right")

    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    pressures1, total_masses1, total_radii1 = ns_calculation(grad_newtonian)
    pressures2, total_masses2, total_radii2 = ns_calculation(grad_TOV)

    # Define line thicknesses for each set of data
    line_thicknesses = [0.75, 2]  # Example line thicknesses for Newtonian and TOV respectively

    plot_mass_radius_vs_pressure([pressures1, pressures2], [total_masses1, total_masses2],
                                 [total_radii1, total_radii2], ['Newtonian', 'TOV'], line_thicknesses, 'log')
