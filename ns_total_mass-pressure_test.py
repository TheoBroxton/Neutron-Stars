"""
Calculate total radius and mass of neutron star for a given initial pressure.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, hbar, G, m_e, m_p, m_n, pi
from scipy.integrate import solve_ivp


# Define parameters
# r=1e-10 to avoid division by zero
initial_state_wd = [1e-10, 2.33002e21, 0.0]
initial_state_ns = [1e-10, 1e32, 0.0]  # r=1e-10 to avoid division by zero
final_time_wd = 1.8e7  # Adjust as needed
final_time_ns = 1e8
step_size_wd = 1e4  # Adjust as needed
step_size_ns = 1e3  # Adjust as needed
M0 = 1.9891e30  # Solar mass in kg
R0 = G * M0 / c**2  # Half Schwarzchild radius in km
gamma_nrel = 5/3
eta_wd = 2  # Ratio of A/Z for white dwarf
eta_ns = 1  # Ratio of A/Z for neutron star
m_N = 0.5 * (m_p + m_n)
K_nrel_wd = ((hbar**2) / (15 * pi**2 * m_e)) * \
    ((3 * pi**2) / (m_N * eta_wd * c**2))**(gamma_nrel)
K_nrel_ns = ((hbar**2) / (15 * pi**2 * m_n)) * \
    ((3 * pi**2) / (m_N * eta_ns * c**2))**(gamma_nrel)


def grad(time, state, M0, gamma_nrel, K_nrel):
    """
    Compute the gradients of the state variables.

    Parameters:
        time (float): Current time.
        state (array-like): Array containing the current values of the state
                            variables [r, p, mbar].
        M0 (float): Solar mass in kg.
        gamma_nrel (float): Adiabatic index for neutron star.
        K_nrel (float): Equation of state constant.

    Returns:
        list: Gradients of the state variables [dr_dt, dp_dt, dmbar_dt].
    """
    r, p, mbar = state

    # Check if pressure is non-positive, return zero gradients if so
    if p <= 0:
        return [0, 0, 0]

    # Compute dp/dr and dmbar/dr
    dp_dr = -(R0 * p**(1/gamma_nrel) * mbar) / \
        (r**2 * K_nrel**(1/gamma_nrel))
    dmbar_dr = (4 * pi * r**2 * p**(1/gamma_nrel)) / \
        (M0 * c**2 * K_nrel**(1/gamma_nrel))

    return [1, dp_dr, dmbar_dr]


def solve_neutron_star_ode(initial_state, final_time, step_size, M0, gamma_nrel, K_nrel):
    sol = solve_ivp(grad, [0, final_time], initial_state, method='RK45', max_step=step_size,
                    args=(M0, gamma_nrel, K_nrel))
    return sol


def plot_neutron_star(sol):
    r_values = sol.y[0]
    pressure_values = sol.y[1]
    mbar_values = sol.y[2]
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

    # Find the index where pressure becomes non-positive
    zero_pressure_index = np.where(pressure_values <= 0)[0][0]

    # Extract the corresponding distance from the center
    zero_pressure_distance = r_values[zero_pressure_index]

    # Calculate the mass at this distance
    zero_pressure_mass = mass_values[zero_pressure_index]

    # Convert mass to solar masses
    zero_pressure_mass_solar = zero_pressure_mass / M0

    print("Distance from center where pressure reaches 0:",
          zero_pressure_distance, "m")
    print("Mass at this distance:", zero_pressure_mass_solar, "solar masses")


# ---- WD ----
# Solve ODE
wd_sol = solve_neutron_star_ode(
    initial_state_wd, final_time_wd, step_size_wd, M0, gamma_nrel, K_nrel_wd)

# Plot results
plot_neutron_star(wd_sol)


# ---- NS ----
# Solve ODE
ns_sol = solve_neutron_star_ode(
    initial_state_ns, final_time_ns, step_size_ns, M0, gamma_nrel, K_nrel_ns)

# Plot results
plot_neutron_star(ns_sol)
