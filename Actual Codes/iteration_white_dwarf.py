# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:38:07 2024

@author: Joshua Edwards
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, hbar, G, m_e, m_p, m_n, pi

# Constants
gamma_nrel = 5/3
M0 = 1.9891e30  # Solar mass in kg
m_N = 0.5 * (m_p + m_n)
eta_wd = 2  # Ratio of A/Z for white dwarf

# Equation of state constant
K_nrel = ((hbar**2) / (15 * pi**2 * m_e)) * \
    ((3 * pi**2) / (m_N * eta_wd * c**2))**(gamma_nrel)

# Compute R0
R0 = (G * M0) / c**2

def rk4_step(grad, time, state, step_size):
    # Calculate various midpoint k states
    k1 = grad(time, state)*step_size
    k2 = grad(time+step_size/2, state+k1/2)*step_size
    k3 = grad(time+step_size/2, state+k2/2)*step_size
    k4 = grad(time+step_size, state+k3)*step_size
    # Return new time and state
    return time+step_size, state+(k1/2 + k2 + k3 + k4/2)/3

def rk4_step_till(grad, time, state, step_size, final_time):
    # Prepare numpy arrays for storing data
    times = np.array([time,])
    state_arr = np.empty(shape=(0, state.size))
    # We will use vstack to add new time slices the state array
    state_arr = np.vstack((state_arr, state))

    # Take as many steps as needed
    while times[-1] < final_time:
        new_time, new_state = rk4_step(
            grad, times[-1], state_arr[-1], step_size)
        times = np.append(times, new_time)
        state_arr = np.vstack((state_arr, new_state))

    return times, state_arr

def grad(time, state):
    r, p, mbar = state

    # Check if pressure is non-positive, return zero gradients if so
    if p <= 0:
        return np.array([0, 0, 0])

    # Compute dp/dr and dmbar/dr
    dp_dr = -(R0 * p**(1/gamma_nrel) * mbar) / (r**2 * K_nrel**(1/gamma_nrel))
    dmbar_dr = (4 * pi * r**2 * p**(1/gamma_nrel)) / \
        (M0 * c**2 * K_nrel**(1/gamma_nrel))

    return np.array([1, dp_dr, dmbar_dr])

# Function to perform integration for a given initial pressure
def integrate(initial_pressure):
    # Define initial state vector
    state0 = np.array([1e-21, initial_pressure, 0.0])  # r=1e-10 to avoid division by zero

    # Specify step size and final time
    step_size = 2e4  # Adjust as needed
    final_time = 1e8  # Adjust as needed

    # Perform RK4 integration
    times, state_arr = rk4_step_till(grad, 0, state0, step_size, final_time)

    # Extract results
    r_values = state_arr[:, 0] /1000
    pressure_values = state_arr[:, 1]
    mbar_values = state_arr[:, 2]

    # Convert mbar_values to mass values
    mass_values = mbar_values * M0

    # Find the index where pressure becomes non-positive
    zero_pressure_index = np.where(pressure_values <= 0)[0][0]

    # Extract the corresponding distance from the center
    zero_pressure_distance = r_values[zero_pressure_index]

    # Calculate the mass at this distance
    zero_pressure_mass = mass_values[zero_pressure_index]

    # Convert mass to solar masses
    zero_pressure_mass_solar = zero_pressure_mass / M0

    return initial_pressure, zero_pressure_distance, zero_pressure_mass_solar

# Array to store results
results = []

# Iterate over initial pressures
for initial_pressure in np.linspace(0, 4.2e21, 100):
    results.append(integrate(initial_pressure))

# Convert results to numpy array
results = np.array(results)

# Plotting
fig, ax1 = plt.subplots()

# Plot radius
color = 'tab:blue'
ax1.set_xlabel('Initial Pressure (Pa)')
ax1.set_ylabel('Radius (km)', color=color)
ax1.plot(results[1:, 0], results[1:, 1], color=color, linestyle = '--', label='Total Radius') # Add label here
ax1.tick_params(axis='y', labelcolor=color)

# Create a second y-axis for mass
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Mass ($M_0$)', color=color)
ax2.plot(results[1:, 0], results[1:, 2], color=color, label='Total Mass') # Add label here
ax2.tick_params(axis='y', labelcolor=color)

# Add legend
ax1.legend(loc='upper left', bbox_to_anchor=(0.08, 0.98))  # Legend for radius
ax2.legend(loc='upper left', bbox_to_anchor=(0.08, 0.88))  # Legend for mass


plt.title('Radius and Mass vs Initial Pressure')
plt.show()

