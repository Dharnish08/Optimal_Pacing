"""
MPC Controller Implementation
==============================

This module implements the Model Predictive Control (MPC) controller
for optimal pacing in swimming competitions using orthogonal collocation.

Author: Advanced Process Control Project
Course: WS 2025/2026
"""

import numpy as np
from casadi import Opti, MX, vertcat
from swimmer_model import SwimmerModel, get_default_parameters
from collocation import CollocationScheme


class SwimmerMPC:
    """
    MPC Controller for optimal swimming pacing.

    Uses orthogonal collocation on finite elements for discretization.
    """

    def __init__(self, N_horizon, dt_control, params=None, cost_type='velocity', d_poly=3):
        """
        Initialize MPC controller.

        Args:
            N_horizon (int): Prediction horizon (number of control intervals)
            dt_control (float): Control interval duration (seconds)
            params (dict): Model parameters (if None, use defaults)
            cost_type (str): Cost function type ('velocity' or 'time')
            d_poly (int): Polynomial degree for collocation
        """
        self.N_horizon = N_horizon
        self.dt_control = dt_control
        self.cost_type = cost_type
        self.d_poly = d_poly

        # Get parameters
        if params is None:
            self.params = get_default_parameters()
        else:
            self.params = params

        # Create model
        self.model = SwimmerModel(self.params)

        # Setup collocation
        self.collocation = CollocationScheme(d_poly=d_poly)

        # Build optimization problem
        self.opti = None
        self.variables = None
        self._build_optimization_problem()

    def _build_optimization_problem(self):
        """Build the MPC optimization problem."""
        self.opti = Opti()

        # Decision variables for states at control intervals
        X = self.opti.variable(self.model.nx, self.N_horizon + 1)
        U = self.opti.variable(self.model.nu, self.N_horizon)

        # Collocation states (interior points)
        X_col = []
        for k in range(self.N_horizon):
            X_col.append(self.opti.variable(self.model.nx, self.d_poly))

        # Initial state parameter
        X0 = self.opti.parameter(self.model.nx, 1)

        # Target distance parameter
        target_distance = self.opti.parameter()

        # Cost function
        cost = 0

        # Initial condition
        self.opti.subject_to(X[:, 0] == X0)

        # Extract collocation matrices
        C = self.collocation.C
        D = self.collocation.D

        # Loop over control intervals
        for k in range(self.N_horizon):
            # Current state and control
            Xk = X[:, k]
            Uk = U[:, k]

            # Add cost
            if self.cost_type == 'velocity':
                # Maximize velocity (minimize negative velocity)
                cost += -X[1, k] * self.dt_control
            elif self.cost_type == 'time':
                # Minimize time
                cost += self.dt_control
            elif self.cost_type == 'energy':
                # Minimize energy usage (minimize force)
                cost += Uk[0]**2 * self.dt_control

            # State at end of interval
            Xk_end = D[0] * Xk

            # Loop over collocation points
            for j in range(1, self.d_poly + 1):
                # Expression for state derivative at collocation point
                xp = C[0, j] * Xk
                for r in range(self.d_poly):
                    xp += C[r + 1, j] * X_col[k][:, r]

                # State at collocation point
                Xc = X_col[k][:, j - 1]

                # Evaluate system dynamics
                fj = self.model.system_fn(Xc, Uk)

                # Collocation equation
                self.opti.subject_to(self.dt_control * fj == xp)

                # Contribution to continuity equation
                Xk_end += D[j] * Xc

            # Continuity equation
            self.opti.subject_to(X[:, k + 1] == Xk_end)

            # Control constraints
            u_force = Uk[0]
            P_anaerobic = Uk[1]
            F_k = X[3, k]  # Fatigue

            # Force constraint: u + alpha*F <= u_max
            self.opti.subject_to(u_force + self.params['alpha'] * F_k <= self.params['umax'])
            self.opti.subject_to(u_force >= 0)

            # Anaerobic power constraint
            self.opti.subject_to(P_anaerobic >= 0)

            # Aerobic power constraint
            v_k = X[1, k]
            E_al_k = X[2, k]

            P_mech_k = u_force * v_k
            P_body_k = P_mech_k / self.params['beta']
            P_al_k = self.params['Pmax_al'] * (E_al_k / self.params['Emax_al'])
            P_aerob_k = P_body_k - P_al_k - P_anaerobic

            self.opti.subject_to(P_aerob_k <= self.params['Pmax_aerobic'])
            self.opti.subject_to(P_aerob_k >= 0)

            # State constraints
            self.opti.subject_to(X[1, k] >= 0)  # velocity >= 0
            self.opti.subject_to(X[2, k] >= 0)  # E_al >= 0
            self.opti.subject_to(X[3, k] >= 0)  # F >= 0

        # Terminal constraint
        self.opti.subject_to(X[0, -1] >= target_distance)

        # Set objective
        self.opti.minimize(cost)

        # Solver options
        opts = {
            'ipopt.print_level': 3,
            'print_time': False,
            'ipopt.max_iter': 3000,
            'ipopt.tol': 1e-6,
            'ipopt.acceptable_tol': 1e-5,
            'ipopt.acceptable_iter': 15
        }
        self.opti.solver('ipopt', opts)

        # Store variables
        self.variables = {
            'X': X,
            'U': U,
            'X0': X0,
            'X_col': X_col,
            'target_distance': target_distance
        }

    def solve(self, x0, target_distance=100.0, initial_guess=None, verbose=True):
        """
        Solve the MPC optimization problem.

        Args:
            x0 (array): Initial state [s, v, E_al, F]
            target_distance (float): Target race distance (m)
            initial_guess (dict): Initial guess for variables (optional)
            verbose (bool): Print solver output

        Returns:
            dict: Solution containing X_opt, U_opt, and race_time
        """
        # Set initial state
        self.opti.set_value(self.variables['X0'], x0)

        # Set target distance
        self.opti.set_value(self.variables['target_distance'], target_distance)

        # Set initial guess
        if initial_guess is None:
            # Default initial guess
            self.opti.set_initial(
                self.variables['X'][0, :],
                np.linspace(0, target_distance, self.N_horizon + 1)
            )
            self.opti.set_initial(self.variables['X'][1, :], 2.0)
            self.opti.set_initial(self.variables['X'][2, :], self.params['Emax_al'])
            self.opti.set_initial(self.variables['X'][3, :], 0.0)
            self.opti.set_initial(self.variables['U'][0, :], 50.0)
            self.opti.set_initial(self.variables['U'][1, :], 100.0)
        else:
            # Use provided initial guess
            if 'X' in initial_guess:
                self.opti.set_initial(self.variables['X'], initial_guess['X'])
            if 'U' in initial_guess:
                self.opti.set_initial(self.variables['U'], initial_guess['U'])

        # Solve
        try:
            if verbose:
                print(f"\nSolving MPC with {self.cost_type} cost function...")
                print(f"Horizon: {self.N_horizon}, dt: {self.dt_control} s")

            sol = self.opti.solve()

            # Extract solution
            X_opt = sol.value(self.variables['X'])
            U_opt = sol.value(self.variables['U'])

            # Calculate race time
            race_time = self.N_horizon * self.dt_control

            if verbose:
                print(f"\nOptimization successful!")
                print(f"Race time: {race_time:.2f} s")
                print(f"Final distance: {X_opt[0, -1]:.2f} m")
                print(f"Final velocity: {X_opt[1, -1]:.2f} m/s")
                print(f"Final fatigue: {X_opt[3, -1]:.2f} mmol/l")

            return {
                'X_opt': X_opt,
                'U_opt': U_opt,
                'race_time': race_time,
                'success': True,
                'solver_stats': sol.stats()
            }

        except Exception as e:
            if verbose:
                print(f"\nOptimization failed: {e}")

            return {
                'X_opt': None,
                'U_opt': None,
                'race_time': None,
                'success': False,
                'error': str(e)
            }

    def get_power_trajectories(self, X_opt, U_opt):
        """
        Calculate power trajectories from optimal solution.

        Args:
            X_opt (array): Optimal state trajectory
            U_opt (array): Optimal control trajectory

        Returns:
            dict: Power trajectories
        """
        N = U_opt.shape[1]

        P_mech = np.zeros(N)
        P_body = np.zeros(N)
        P_al = np.zeros(N)
        P_aerob = np.zeros(N)
        P_anaerobic = np.zeros(N)

        for k in range(N):
            x_k = X_opt[:, k]
            u_k = U_opt[:, k]
            powers = self.model.get_power_components(x_k, u_k)

            P_mech[k] = powers['P_mech']
            P_body[k] = powers['P_body']
            P_al[k] = powers['P_al']
            P_aerob[k] = powers['P_aerob']
            P_anaerobic[k] = powers['P_anaerobic']

        return {
            'P_mech': P_mech,
            'P_body': P_body,
            'P_al': P_al,
            'P_aerob': P_aerob,
            'P_anaerobic': P_anaerobic
        }


def compare_cost_functions(N_horizon=50, dt_control=0.5):
    """
    Compare different cost functions.

    Args:
        N_horizon (int): Prediction horizon
        dt_control (float): Control interval
    """
    print("\n" + "="*60)
    print("COMPARING COST FUNCTIONS")
    print("="*60)

    # Initial state
    params = get_default_parameters()
    x0 = np.array([0.0, 3.0, params['Emax_al'], 0.0])

    cost_types = ['velocity', 'time']
    results = {}

    for cost_type in cost_types:
        print(f"\n{'='*60}")
        print(f"Cost Function: {cost_type.upper()}")
        print(f"{'='*60}")

        mpc = SwimmerMPC(
            N_horizon=N_horizon,
            dt_control=dt_control,
            cost_type=cost_type
        )

        result = mpc.solve(x0, target_distance=100.0, verbose=True)
        results[cost_type] = result

    # Print comparison
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)

    for cost_type, result in results.items():
        if result['success']:
            print(f"\n{cost_type.upper()}:")
            print(f"  Race time: {result['race_time']:.2f} s")
            print(f"  Avg velocity: {100.0/result['race_time']:.2f} m/s")
            print(f"  Final fatigue: {result['X_opt'][3, -1]:.2f} mmol/l")
            print(f"  Final energy: {result['X_opt'][2, -1]:.2f} J")

    return results


if __name__ == "__main__":
    """Test the MPC controller."""
    print("MPC Controller Test")
    print("="*60)

    # Test single optimization
    params = get_default_parameters()
    x0 = np.array([0.0, 3.0, params['Emax_al'], 0.0])

    mpc = SwimmerMPC(
        N_horizon=50,
        dt_control=0.5,
        cost_type='velocity'
    )

    result = mpc.solve(x0, target_distance=100.0, verbose=True)

    if result['success']:
        print("\nMPC optimization successful!")

        # Calculate power trajectories
        powers = mpc.get_power_trajectories(
            result['X_opt'],
            result['U_opt']
        )

        print(f"\nAverage powers:")
        for key, values in powers.items():
            print(f"  {key}: {np.mean(values):.2f} W")

    # Compare cost functions
    print("\n" + "="*60)
    compare_cost_functions()

    print("\n" + "="*60)
    print("MPC controller test completed!")
