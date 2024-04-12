import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, hbar, G, m_e, m_p, m_n, pi
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

def p_x(x, epsilon0):
    """
    Compute the function p(x) using the given equation.

    Parameters:
        x (float): Value of x.
        epsilon0 (float): Constant value.

    Returns:
        float: Value of p(x) for the given x.
    """
    return (epsilon0 / 24) * ((2 * x**3 - 3 * x) * (1 + x**2)**0.5 + 3 * np.arcsinh(x))

def find_x_for_given_p(p, epsilon0):
    """
    Find the value of x for a given p using root finding.

    Parameters:
        p (float): Value of pressure.
        epsilon0 (float): Constant value.

    Returns:
        float: Value of x for the given pressure.
    """
    func = lambda x: p_x(x, epsilon0) - p
    sol = root_scalar(func, bracket=[0, 1e10])  # Adjust bracket as needed
    return sol.root if sol.converged else None

def epsilon_x(x, epsilon0):
    """
    Compute the function epsilon(x) using the given equation.

    Parameters:
        x (float): Value of x.
        epsilon0 (float): Constant value.

    Returns:
        float: Value of epsilon(x) for the given x.
    """
    return (epsilon0 / 8) * ((2 * x**3 + x) * (1 + x**2)**0.5 - np.arcsinh(x))

def grad(time, state, M0, gamma_nrel, K_nrel, epsilon0, epsilon_values):
    """
    Compute the gradients of the state variables.

    Parameters:
        time (float): Current time.
        state (array-like): Array containing the current values of the state
                            variables [r, p, mbar].
        M0 (float): Solar mass in kg.
        gamma_nrel (float): Adiabatic index for neutron star.
        K_nrel (float): Equation of state constant.
        epsilon0 (float): Constant value.
        epsilon_values (list): List to store epsilon values.

    Returns:
        list: Gradients of the state variables [dr_dt, dp_dt, dmbar_dt].
    """
    r, p, mbar = state

    # Check if pressure is non-positive, return zero gradients if so
    if p <= 0:
        return [0, 0, 0]

    x = find_x_for_given_p(p, epsilon0)
    if x is None:
        return [0, 0, 0]

    epsilon = epsilon_x(x, epsilon0)
    epsilon_values.append(epsilon)  # Store epsilon value
    m = mbar * M0

    # Check for division by zero or invalid values
    if epsilon == 0:
        dp_dr = 0
    else:
        sqrt_value = max(0, 1 - ((2 * G * m) / (c**2 * r)))
        dp_dr = -(((G * epsilon * m) / (c**2 * r**2)) + ((4 * pi * r * p * epsilon * G) / (c**4))) * (1 +(p/epsilon)) * np.sqrt(sqrt_value)**-1
    
    dmbar_dr = (4 * pi * r**2 * p**(1/gamma_nrel)) / \
        (M0 * c**2 * K_nrel**(1/gamma_nrel))

    return [1, dp_dr, dmbar_dr]

def solve_neutron_star_ode(initial_states, final_time, step_size, M0,
                           gamma_nrel, K_nrel, epsilon0, epsilon_values):
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
        epsilon0 (float): Constant value for epsilon.
        epsilon_values (list): List to store epsilon values.

    Returns:
        list of scipy.integrate.OdeSolution: List of solution objects
        containing the integrated results.
    """
    solutions = []
    for initial_state in initial_states:
        sol = solve_ivp(grad, [0, final_time], initial_state, method='RK45',
                        max_step=step_size, args=(M0, gamma_nrel, K_nrel, epsilon0, epsilon_values))
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
    ax1.plot(pressures, total_radii, color=color, label='Total Radius')
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

def ns_calculation():
    # Generate logarithmically spaced initial pressures
    initial_pressures_ns = np.logspace(np.log10(1e30), np.log10(4e41), 7)

    # Define parameters
    M0 = 1.9891e30  # Solar mass in kg
    gamma_nrel = 5/3
    epsilon0 = (m_n**4 * c**5) / (np.pi**2 * hbar**3)
    K_nrel = ((hbar**2) / (15 * pi**2 * m_n)) * ((3 * pi**2) / (m_N * c**2))**(gamma_nrel)
    
    # Empty list to store epsilon values
    epsilon_values = []

    initial_states_ns = [[1e-10, p0, 0.0] for p0 in initial_pressures_ns]
    final_time_ns = 1e7  # Increase the final time for integration
    step_size_ns = 1e5  # Adjust as needed

    # Solve ODE for multiple initial states
    solutions_ns = solve_neutron_star_ode(
        initial_states_ns, final_time_ns, step_size_ns, M0, gamma_nrel,
        K_nrel, epsilon0, epsilon_values)

    # Calculate total mass and total radius for each solution
    total_masses_ns, total_radii_ns = calculate_total_mass_radius(
        solutions_ns, M0)

    # Plot total mass and total radius vs. initial pressure
    plot_mass_radius_vs_pressure(initial_pressures_ns, total_masses_ns,
                                 total_radii_ns, 'NS_varying_p0', 'log')
    
    # Print or visualize epsilon values
    print("Epsilon values:", epsilon_values)

if __name__ == "__main__":
    # Define parameters
    M0 = 1.9891e30  # Solar mass in kg
    R0 = G * M0 / c**2  # Half Schwarzchild radius in km
    gamma_nrel = 5/3
    gamma_rel = 4/3
    m_N = 0.5 * (m_p + m_n)

# Call the ns_calculation function
ns_calculation()
