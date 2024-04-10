import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, hbar, G, m_e, m_p, m_n, pi
from scipy.integrate import solve_ivp

# Define parameters
M0 = 1.9891e30  # Solar mass in kg
R0 = G * M0 / c**2  # Half Schwarzchild radius in km
gamma_nrel = 5/3
gamma_rel = 4/3
m_N = 0.5 * (m_p + m_n)

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

    epsilon = (p/K_nrel)**(1/gamma_nrel)
    m = mbar * M0

    # Compute dp/dr and dmbar/dr
    dp_dr = -(((G * epsilon * m) / (c**2 * r**2)) + ((4 * pi * r * p * epsilon * G) / (c**4))) * (1 +(p/epsilon)) * (1 - ((2 * G * m) / (c**2 * r)))**-1
    
    dmbar_dr = (4 * pi * r**2 * p**(1/gamma_nrel)) / \
        (M0 * c**2 * K_nrel**(1/gamma_nrel))

    return [1, dp_dr, dmbar_dr]


def solve_neutron_star_ode(initial_states, final_time, step_size, M0,
                           gamma_nrel, K_nrel):
    """
    Solves the ODE system for neutron stars for multiple initial states.

    Parameters:
        initial_states (list of array-like): List of initial state vectors
                                             [r0, p0, mbar0].
        final_time (float): Final time for integration.
        step_size (float): Step size for integration.
        M0 (float): Solar mass in kg.
        gamma_nrel (float): Adiabatic index (non-relativistic).
        K_nrel (float): Equation of state constant.

    Returns:
        list of scipy.integrate.OdeSolution: List of solution objects
        containing the integrated results.
    """
    solutions = []
    for initial_state in initial_states:
        sol = solve_ivp(grad, [0, final_time], initial_state, method='RK45',
                        max_step=step_size, args=(M0, gamma_nrel, K_nrel))
        solutions.append(sol)
    return solutions


def calculate_total_mass_radius(solutions, M0):
    """
    Calculate the total mass and total radius for each solution.

    Parameters:
        solutions (list of scipy.integrate.OdeSolution): List of solution
                                                         objects containing the
                                                         integrated results.
        M0 (float): Solar mass in kg.

    Returns:
        tuple of lists: Lists of total mass and total radius for each solution.
    """
    total_masses = []
    total_radii = []
    for sol in solutions:
        radius_values = sol.y[0]
        pressure_values = sol.y[1]
        mbar_values = sol.y[2]
        try:
            # Find the index where pressure becomes non-positive
            zero_pressure_index = np.where(pressure_values <= 0)[0][0]

            # Extract the corresponding mass value
            zero_pressure_mass = mbar_values[zero_pressure_index]

            # Extract the corresponding radius value
            zero_pressure_radius = radius_values[zero_pressure_index]

            total_masses.append(zero_pressure_mass)
            total_radii.append(zero_pressure_radius)
        except IndexError:
            # Handle the case where pressure remains positive throughout
            total_masses.append(np.nan)
            total_radii.append(np.nan)
    return total_masses, total_radii


def grad1(time, state, M0, gamma_nrel, K_nrel):
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


def solve_neutron_star_ode1(initial_states, final_time, step_size, M0,
                           gamma_nrel, K_nrel):
    """
    Solves the ODE system for neutron stars for multiple initial states.

    Parameters:
        initial_states (list of array-like): List of initial state vectors
                                             [r0, p0, mbar0].
        final_time (float): Final time for integration.
        step_size (float): Step size for integration.
        M0 (float): Solar mass in kg.
        gamma_nrel (float): Adiabatic index (non-relativistic).
        K_nrel (float): Equation of state constant.

    Returns:
        list of scipy.integrate.OdeSolution: List of solution objects
        containing the integrated results.
    """
    solutions = []
    for initial_state in initial_states:
        sol = solve_ivp(grad1, [0, final_time], initial_state, method='RK45',
                        max_step=step_size, args=(M0, gamma_nrel, K_nrel))
        solutions.append(sol)
    return solutions


def calculate_total_mass_radius1(solutions, M0):
    """
    Calculate the total mass and total radius for each solution.

    Parameters:
        solutions (list of scipy.integrate.OdeSolution): List of solution
                                                         objects containing the
                                                         integrated results.
        M0 (float): Solar mass in kg.

    Returns:
        tuple of lists: Lists of total mass and total radius for each solution.
    """
    total_masses = []
    total_radii = []
    for sol in solutions:
        radius_values = sol.y[0]
        pressure_values = sol.y[1]
        mbar_values = sol.y[2]
        try:
            # Find the index where pressure becomes non-positive
            zero_pressure_index = np.where(pressure_values <= 0)[0][0]

            # Extract the corresponding mass value
            zero_pressure_mass = mbar_values[zero_pressure_index]

            # Extract the corresponding radius value
            zero_pressure_radius = radius_values[zero_pressure_index]

            total_masses.append(zero_pressure_mass)
            total_radii.append(zero_pressure_radius)
        except IndexError:
            # Handle the case where pressure remains positive throughout
            total_masses.append(np.nan)
            total_radii.append(np.nan)
    return total_masses, total_radii


def ns_calculation():
    # Generate logarithmically spaced initial pressures
    initial_pressures_ns = np.logspace(np.log10(1e30), np.log10(2e32), 7)

    # Rest of the code remains the same
    initial_states_ns = [[1e-10, p0, 0.0] for p0 in initial_pressures_ns]
    final_time_ns = 1e7  # Increase the final time for integration
    step_size_ns = 1e5  # Adjust as needed
    eta_ns = 1  # Ratio of A/Z for neutron star
    K_nrel_ns = ((hbar**2) / (15 * pi**2 * m_n)) * \
        ((3 * pi**2) / (m_N * eta_ns * c**2))**(gamma_nrel)

    # Solve ODE for multiple initial states
    solutions_ns = solve_neutron_star_ode(
        initial_states_ns, final_time_ns, step_size_ns, M0, gamma_nrel,
        K_nrel_ns)

    # Calculate total mass and total radius for each solution
    total_masses_ns, total_radii_ns = calculate_total_mass_radius(
        solutions_ns, M0)

    return initial_pressures_ns, total_masses_ns, total_radii_ns

def ns_calculation1():
    # Generate logarithmically spaced initial pressures
    initial_pressures_ns = np.logspace(np.log10(1e30), np.log10(2e32), 8)

    # Rest of the code remains the same
    initial_states_ns = [[1e-10, p0, 0.0] for p0 in initial_pressures_ns]
    final_time_ns = 1e7  # Increase the final time for integration
    step_size_ns = 1e16  # Adjust as needed
    eta_ns = 1  # Ratio of A/Z for neutron star
    K_nrel_ns = ((hbar**2) / (15 * pi**2 * m_n)) * \
        ((3 * pi**2) / (m_N * eta_ns * c**2))**(gamma_nrel)

    # Solve ODE for multiple initial states
    solutions_ns = solve_neutron_star_ode1(
        initial_states_ns, final_time_ns, step_size_ns, M0, gamma_nrel,
        K_nrel_ns)

    # Calculate total mass and total radius for each solution
    total_masses_ns, total_radii_ns = calculate_total_mass_radius1(
        solutions_ns, M0)

    return initial_pressures_ns, total_masses_ns, total_radii_ns

if __name__ == "__main__":
    # Call the calculation functions to obtain data
    initial_pressures_ns, total_masses_ns, total_radii_ns = ns_calculation()
    initial_pressures_ns1, total_masses_ns1, total_radii_ns1 = ns_calculation1()

    # Plot the results
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot total radius on the left y-axis
    color = 'tab:red'
    ax1.set_xlabel('Initial Pressures (Pa)')
    ax1.set_ylabel('Total Radius (m)', color=color)
    ax1.plot(initial_pressures_ns, total_radii_ns, label='Total Radius - Original', color=color)
    ax1.plot(initial_pressures_ns1, total_radii_ns1, label='Total Radius - Modified', color='orange')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xscale('log')

    # Plot total mass on the right y-axis
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Total Mass ($M_0$)', color=color)  
    ax2.plot(initial_pressures_ns, total_masses_ns, label='Total Mass - Original', color=color)
    ax2.plot(initial_pressures_ns1, total_masses_ns1, label='Total Mass - Modified', color='green')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_xscale('log')

    fig.tight_layout()  
    fig.legend(loc="upper right")  
    plt.show()