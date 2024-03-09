import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, hbar, G, m_e, m_p, m_n, pi
from scipy.integrate import solve_ivp


def grad(time, state, M0, gamma_nrel, K_nrel):
    """
    Computes the gradients of the state variables.

    Parameters:
        time (float): Current time.
        state (array-like): Array containing the current values of the state variables [r, p, mbar].
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


def solve_neutron_star_ode(initial_states, final_time, step_size, M0, gamma_nrel, K_nrel):
    """
    Solves the ODE system for neutron stars for multiple initial states.

    Parameters:
        initial_states (list of array-like): List of initial state vectors [r0, p0, mbar0].
        final_time (float): Final time for integration.
        step_size (float): Step size for integration.
        M0 (float): Solar mass in kg.
        gamma_nrel (float): Adiabatic index (non-relativistic).
        K_nrel (float): Equation of state constant.

    Returns:
        list of scipy.integrate.OdeSolution: List of solution objects containing the integrated results.
    """
    solutions = []
    for initial_state in initial_states:
        sol = solve_ivp(grad, [0, final_time], initial_state, method='RK45', max_step=step_size,
                        args=(M0, gamma_nrel, K_nrel))
        solutions.append(sol)
    return solutions


def calculate_total_mass_radius(solutions, M0):
    """
    Calculates the total mass and total radius for each solution.

    Parameters:
        solutions (list of scipy.integrate.OdeSolution): List of solution objects containing the integrated results.
        M0 (float): Solar mass in kg.

    Returns:
        tuple of lists: Lists of total mass and total radius for each solution.
    """
    total_masses = []
    total_radii = []
    for sol in solutions:
        pressure_values = sol.y[1]
        mbar_values = sol.y[2]

        # Find the index where pressure becomes non-positive
        zero_pressure_index = np.where(pressure_values <= 0)[0][0]

        # Extract the corresponding mass value
        zero_pressure_mass = mbar_values[zero_pressure_index]

        # Extract the corresponding radius value
        zero_pressure_radius = sol.y[0][zero_pressure_index]

        total_masses.append(zero_pressure_mass)
        total_radii.append(zero_pressure_radius)
    return total_masses, total_radii


def plot_mass_radius_vs_pressure(pressures, total_masses, total_radii):
    """
    Plots the total mass and total radius of neutron stars against the initial pressure.

    Parameters:
        pressures (array-like): Array of initial pressures.
        total_masses (array-like): Array of total masses corresponding to each initial pressure.
        total_radii (array-like): Array of total radii corresponding to each initial pressure.
    """
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_ylabel('Total Radius (m)', color=color)
    ax1.set_xlabel('Initial Pressure (Pa)')
    ax1.plot(pressures, total_radii, color=color, label='Total Radius')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Total Mass ($M_0$)', color=color)
    ax2.plot(pressures, total_masses, color=color, label='Total Mass')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    fig.legend(loc="upper right")

    plt.savefig('Figures/WD_varying_p0.pdf', dpi=300)
    plt.show()


if __name__ == "__main__":
    # Define parameters
    initial_pressures = np.linspace(1e20, 5e22, 100)
    initial_states = [[1e-10, p0, 0.0] for p0 in initial_pressures]
    final_time = 1.8e7  # Adjust as needed
    step_size = 1e4  # Adjust as needed
    M0 = 1.9891e30  # Solar mass in kg
    R0 = G * M0 / c**2  # Half Schwarzchild radius in km
    gamma_nrel = 5/3
    eta_wd = 2  # Ratio of A/Z for white dwarf
    m_N = 0.5 * (m_p + m_n)
    K_nrel = ((hbar**2) / (15 * pi**2 * m_e)) * \
        ((3 * pi**2) / (m_N * eta_wd * c**2))**(gamma_nrel)

    # Solve ODE for multiple initial states
    solutions = solve_neutron_star_ode(
        initial_states, final_time, step_size, M0, gamma_nrel, K_nrel)

    # Calculate total mass and total radius for each solution
    total_masses, total_radii = calculate_total_mass_radius(solutions, M0)

    # Plot total mass and total radius vs. initial pressure
    plot_mass_radius_vs_pressure(initial_pressures, total_masses, total_radii)
