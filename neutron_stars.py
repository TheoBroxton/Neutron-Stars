# -*- coding: utf-8 -*-
"""
--------------TITLE--------------.

PHYS20872 Project - Neutron Stars
---------------------------------
This Python script performs calculations based on neutron star parameters.

Last updated: 27/02/24
Authors: Theo Broxton, Joshua Edwards
"""
import numpy as np
import matplotlib.pyplot as plt
# import scipy.constants as pc
from scipy.constants import c, hbar, G, m_e, m_p, m_n, pi

# Constants
gamma_nrel = 5/3
gamma_rel = 4/3
# G = 6.67430e-11  # Gravitational constant in m^3/kg/s^2
# c = 299792458  # Speed of light in m/s
M0 = 1.9891e30  # Solar mass in kg
# hbar = 1.0545718e-34  # Planck constant over 2*pi in J*s
# mn = 1.674927471e-27  # Mass of a neutron in kg
# me = 9.1093837015e-31  # Mass of an electron in kg
m_N = 0.5 * (m_p + m_n)
eta_wd = 2  # Ratio of A/Z for white dwarf
eta_n = 1  # Ratio of A/Z for neutron star

# Equation of state constant
K_nrel = ((hbar**2) / (15 * pi**2 * m_e)) * \
    ((3 * pi**2) / (m_N * eta_wd * c**2))**(gamma_nrel)
K_rel = ((hbar * c) / (12 * pi**2)) * \
    ((3 * pi**2) / (m_N * eta_wd * c**2))**(gamma_rel)

# Epsilon_0 for a Fermi gas
epsilon0 = (m_n**4 * c**5) / (pi**2 * hbar**3)

# Compute R0
R0 = G * M0 / c**2

# Initial conditions
p0 = 2.33002e21  # Initial pressure at r=0 in Pa
mbar0 = 0.0  # Initial mass at r=0 (normalized) in kg/M0

# Function for taking a single RK4 step
# The grad function must be of form grad(time (float), state (numpy array of floats)) -> (numpy array of floats)


def rk4_step(grad, time, state, step_size):
    # Calculate various midpoint k states
    k1 = grad(time, state)*step_size
    k2 = grad(time+step_size/2, state+k1/2)*step_size
    k3 = grad(time+step_size/2, state+k2/2)*step_size
    k4 = grad(time+step_size, state+k3)*step_size
    # Return new time and state
    return time+step_size, state+(k1/2 + k2 + k3 + k4/2)/3

# Function for taking n steps using RK4


def rk4_n_steps(grad, time, state, step_size, n_steps):
    # Prepare numpy arrays for storing data
    times = np.array([time,])
    state_arr = np.empty(shape=(0, state.size))
    # We will use vstack to add new time slices the state array
    state_arr = np.vstack((state_arr, state))

    # Take n steps
    for _ in range(n_steps):
        new_time, new_state = rk4_step(
            grad, times[-1], state_arr[-1], step_size)
        times = np.append(times, new_time)
        state_arr = np.vstack((state_arr, new_state))

    return times, state_arr

# Function for taking steps till some final time using RK4. Useful for comparing step sizes


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

    # epsilon = (epsilon0 / 3) * r**3

    # Compute epsilon from the polytropic equation of state
    # p = K*epsilon**gamma

    # Compute dp/dr and dmbar/dr
    dp_dr = -(R0 * p**(1/gamma_nrel) * mbar) / (r**2 * K_nrel**(1/gamma_nrel))
    dmbar_dr = (4 * pi * r**2 * p**(1/gamma_nrel)) / \
        (M0 * c**2 * K_nrel**(1/gamma_nrel))

    return np.array([1, dp_dr, dmbar_dr])


# Define initial state vector
state0 = np.array([1e-10, p0, mbar0])  # r=1e-10 to avoid division by zero

# Specify step size and final time
step_size = 1e5  # Adjust as needed
final_time = 1e8  # Adjust as needed

# Perform RK4 integration
times, state_arr = rk4_step_till(grad, 0, state0, step_size, final_time)

# Extract results
r_values = state_arr[:, 0]
pressure_values = state_arr[:, 1]
mbar_values = state_arr[:, 2]

# Convert mbar_values to mass values
mass_values = mbar_values * M0

fig, ax1 = plt.subplots()

# Plot pressure on the left y-axis
color = 'tab:blue'
ax1.set_xlabel('Distance from Center (m)')
ax1.set_ylabel('Pressure (Pa)', color=color)
ax1.plot(r_values, pressure_values, color=color, label='Pressure')
ax1.tick_params(axis='y', labelcolor=color)

# Create a second y-axis for mass
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Mass ($M_0$)', color=color)
ax2.plot(r_values, mass_values / M0, color=color, label='Mass ($M_0$)')
ax2.tick_params(axis='y', labelcolor=color)

# Show legend
fig.tight_layout()
fig.legend(loc="upper right")

plt.savefig('Figures/WD_non-rel.pdf', dpi=300)
plt.show()
