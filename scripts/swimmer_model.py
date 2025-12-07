"""
Swimmer Model Implementation
=============================

This module implements the swimmer dynamics model including:
- Mechanics (force balance)
- Energy systems (alactacid, aerobic, anaerobic)
- Fatigue dynamics
- Power balance equations

Author: Advanced Process Control Project
Course: WS 2025/2026
"""

import numpy as np
from casadi import MX, vertcat, Function


class SwimmerModel:
    """
    Complete swimmer model for optimal pacing.

    States: [s, v, E_al, F]
    - s: distance (m)
    - v: velocity (m/s)
    - E_al: alactacid energy (J)
    - F: fatigue (mmol/l)

    Controls: [u, P_anaerobic]
    - u: propulsion force (N)
    - P_anaerobic: anaerobic power (W)
    """

    def __init__(self, params):
        """
        Initialize swimmer model with parameters.

        Args:
            params (dict): Model parameters
        """
        self.params = params
        self.nx = 4  # Number of states
        self.nu = 2  # Number of controls

        # Create CasADi function
        self.system_fn = self._create_system_function()

    def _create_system_function(self):
        """
        Create the system dynamics as a CasADi function.

        Returns:
            Function: CasADi function mapping (x, u) -> xdot
        """
        # State variables
        s = MX.sym('s')         # distance
        v = MX.sym('v')         # velocity
        E_al = MX.sym('E_al')   # alactacid energy
        F = MX.sym('F')         # fatigue

        x = vertcat(s, v, E_al, F)

        # Control variables
        u_force = MX.sym('u')              # propulsion force
        P_anaerobic = MX.sym('P_anaerobic') # anaerobic power

        u = vertcat(u_force, P_anaerobic)

        # Extract parameters
        gamma = self.params['gamma']
        cw = self.params['cw']
        A = self.params['A']
        beta = self.params['beta']
        m = self.params['m']
        d = self.params['d']
        Pmax_al = self.params['Pmax_al']
        Emax_al = self.params['Emax_al']
        rho = self.params['rho']

        # Mechanical power
        P_mech = u_force * v

        # Body power required
        P_body = P_mech / beta

        # Alactacid power (proportional to remaining energy)
        P_al = Pmax_al * (E_al / Emax_al)

        # Aerobic power (remaining power after alactacid and anaerobic)
        P_aerob = P_body - P_al - P_anaerobic

        # System dynamics
        s_dot = v
        v_dot = (gamma * u_force - 0.5 * rho * cw * A * v**2) / m
        E_al_dot = -P_al
        F_dot = -P_anaerobic / d

        xdot = vertcat(s_dot, v_dot, E_al_dot, F_dot)

        # Create CasADi function
        system = Function('system', [x, u], [xdot], ['x', 'u'], ['xdot'])

        return system

    def evaluate(self, x, u):
        """
        Evaluate system dynamics at given state and control.

        Args:
            x (array): State vector [s, v, E_al, F]
            u (array): Control vector [u_force, P_anaerobic]

        Returns:
            array: State derivative xdot
        """
        xdot = self.system_fn(x, u)
        return np.array(xdot).flatten()

    def get_power_components(self, x, u):
        """
        Calculate all power components.

        Args:
            x (array): State vector
            u (array): Control vector

        Returns:
            dict: Power components (P_mech, P_body, P_al, P_aerob, P_anaerobic)
        """
        v = x[1]
        E_al = x[2]
        u_force = u[0]
        P_anaerobic = u[1]

        # Calculate power components
        P_mech = u_force * v
        P_body = P_mech / self.params['beta']
        P_al = self.params['Pmax_al'] * (E_al / self.params['Emax_al'])
        P_aerob = P_body - P_al - P_anaerobic

        return {
            'P_mech': P_mech,
            'P_body': P_body,
            'P_al': P_al,
            'P_aerob': P_aerob,
            'P_anaerobic': P_anaerobic
        }

    def check_constraints(self, x, u):
        """
        Check if state and control satisfy constraints.

        Args:
            x (array): State vector
            u (array): Control vector

        Returns:
            dict: Constraint violations
        """
        violations = {}

        # Extract variables
        v = x[1]
        E_al = x[2]
        F = x[3]
        u_force = u[0]
        P_anaerobic = u[1]

        # Check force constraint: u + alpha*F <= u_max
        force_with_fatigue = u_force + self.params['alpha'] * F
        if force_with_fatigue > self.params['umax']:
            violations['force'] = force_with_fatigue - self.params['umax']

        # Check power constraints
        powers = self.get_power_components(x, u)

        if powers['P_aerob'] > self.params['Pmax_aerobic']:
            violations['aerobic_power'] = powers['P_aerob'] - self.params['Pmax_aerobic']

        if powers['P_aerob'] < 0:
            violations['negative_aerobic'] = -powers['P_aerob']

        if P_anaerobic < 0:
            violations['negative_anaerobic'] = -P_anaerobic

        if u_force < 0:
            violations['negative_force'] = -u_force

        if v < 0:
            violations['negative_velocity'] = -v

        if E_al < 0:
            violations['negative_energy'] = -E_al

        return violations


def get_default_parameters():
    """
    Get default model parameters from the project specification.

    Returns:
        dict: Model parameters
    """
    params = {
        'gamma': 0.5,              # Force conversion efficiency
        'cw': 2.0,                 # Drag coefficient
        'A': 0.02,                 # Body area (m²)
        'beta': 0.2,               # Body efficiency
        'm': 70.0,                 # Mass (kg)
        'd': 5000.0,               # Energy to fatigue conversion (kJ/(mmol/l))
        'Pmax_al': 1000.0,         # Max alactacid power (W)
        'Pmax_aerobic': 300.0,     # Max aerobic power (W)
        'Emax_al': 1000.0,         # Max alactacid energy (J)
        'rho': 997.0,              # Water density (kg/m³)
        'umax': 200.0,             # Max force (N)
        'alpha': 10.0,             # Fatigue to force reduction (N/(mmol/l))
    }
    return params


if __name__ == "__main__":
    """Test the swimmer model."""
    print("Testing Swimmer Model")
    print("=" * 60)

    # Get parameters
    params = get_default_parameters()

    # Create model
    model = SwimmerModel(params)

    print(f"\nModel created successfully!")
    print(f"Number of states: {model.nx}")
    print(f"Number of controls: {model.nu}")

    # Test evaluation
    x_test = np.array([0.0, 2.0, 1000.0, 0.0])  # [s, v, E_al, F]
    u_test = np.array([50.0, 100.0])            # [u, P_anaerobic]

    print(f"\nTest evaluation:")
    print(f"State: {x_test}")
    print(f"Control: {u_test}")

    xdot = model.evaluate(x_test, u_test)
    print(f"State derivative: {xdot}")

    powers = model.get_power_components(x_test, u_test)
    print(f"\nPower components:")
    for key, value in powers.items():
        print(f"  {key}: {value:.2f} W")

    violations = model.check_constraints(x_test, u_test)
    if violations:
        print(f"\nConstraint violations:")
        for key, value in violations.items():
            print(f"  {key}: {value:.2f}")
    else:
        print(f"\nAll constraints satisfied!")

    print("\n" + "=" * 60)
    print("Swimmer model test completed successfully!")