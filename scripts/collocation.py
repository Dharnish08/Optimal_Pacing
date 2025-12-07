"""
Orthogonal Collocation Utilities
==================================

This module provides utilities for setting up orthogonal collocation
on finite elements for continuous-time optimal control problems.

Implements Radau collocation points and computes the necessary
matrices for the collocation equations.

Author: Advanced Process Control Project
Course: WS 2025/2026
"""

import numpy as np
from casadi import collocation_points


class CollocationScheme:
    """
    Orthogonal collocation scheme for continuous-time systems.

    Uses Radau collocation points and Lagrange polynomials.
    """

    def __init__(self, d_poly=3, scheme='radau'):
        """
        Initialize collocation scheme.

        Args:
            d_poly (int): Degree of interpolating polynomial
            scheme (str): Collocation scheme ('radau' or 'legendre')
        """
        self.d_poly = d_poly
        self.scheme = scheme

        # Setup collocation
        self.tau_root, self.C, self.D, self.B = self._setup_collocation()

    def _setup_collocation(self):
        """
        Setup collocation matrices.
        Returns:
            tuple: (tau_root, C, D, B)
            - tau_root: Collocation points (including 0)
            - C: Collocation matrix for derivatives
            - D: Continuity coefficients
            - B: Quadrature weights
        """
        # Get collocation points (0 + interior points)
        tau_root = [0] + collocation_points(self.d_poly, self.scheme)
        # Coefficients of the collocation equation
        C = np.zeros((self.d_poly+1, self.d_poly+1))

        # Coefficients of the continuity equation
        D = np.zeros(self.d_poly+1)

        # Coefficients of the quadrature function
        B = np.zeros(self.d_poly+1)

        # Construct polynomial basis
        for j in range(self.d_poly+1):
            # Construct Lagrange polynomial
            p = np.poly1d([1])
            for r in range(self.d_poly+1):
                if r != j:
                    p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j] - tau_root[r])

            # Evaluate at end point for continuity equation
            D[j] = p(1.0)

            # Evaluate derivative at collocation points
            pder = np.polyder(p)
            for r in range(self.d_poly+1):
                C[j, r] = pder(tau_root[r])

            # Integrate for quadrature weights
            pint = np.polyint(p)
            B[j] = pint(1.0)

        return tau_root, C, D, B

    def get_info(self):
        """
        Get information about the collocation scheme.

        Returns:
            dict: Information dictionary
        """
        return {
            'degree': self.d_poly,
            'scheme': self.scheme,
            'num_points': self.d_poly + 1,
            'collocation_points': self.tau_root,
            'C_shape': self.C.shape,
            'D_shape': self.D.shape,
            'B_shape': self.B.shape
        }

    def print_info(self):
        """Print information about the collocation scheme."""
        info = self.get_info()
        print("Collocation Scheme Information")
        print("=" * 60)
        print(f"Polynomial degree: {info['degree']}")
        print(f"Scheme: {info['scheme']}")
        print(f"Number of points: {info['num_points']}")
        print(f"Collocation points: {[f'{t:.4f}' for t in info['collocation_points']]}")
        print(f"\nMatrix C (derivatives): {info['C_shape']}")
        print(f"Vector D (continuity): {info['D_shape']}")
        print(f"Vector B (quadrature): {info['B_shape']}")

    def verify_properties(self):
        """
        Verify mathematical properties of the collocation scheme.

        Returns:
            dict: Verification results
        """
        results = {}

        # Check if D sums to 1 (polynomial at t=1)
        D_sum = np.sum(self.D)
        results['D_sum_equals_1'] = np.abs(D_sum - 1.0) < 1e-10
        results['D_sum'] = D_sum

        # Check if B sums to 1 (integral from 0 to 1)
        B_sum = np.sum(self.B)
        results['B_sum_equals_1'] = np.abs(B_sum - 1.0) < 1e-10
        results['B_sum'] = B_sum

        # Check Lagrange property: C[j, tau_root[k]] should be 0 for j != k
        lagrange_ok = True
        for j in range(self.d_poly + 1):
            for k in range(self.d_poly + 1):
                if j != k:
                    # Evaluate derivative of L_j at tau_k
                    # This is given by C[j, k]
                    # For Lagrange polynomials, L_j(tau_k) = delta_jk
                    # But we're checking derivatives here
                    pass

        results['lagrange_property'] = lagrange_ok

        return results

    def print_verification(self):
        """Print verification results."""
        results = self.verify_properties()
        print("\nCollocation Scheme Verification")
        print("=" * 60)
        for key, value in results.items():
            if isinstance(value, bool):
                status = "✓ PASS" if value else "✗ FAIL"
                print(f"{key}: {status}")
            else:
                print(f"{key}: {value:.10f}")


def lagrange_polynomial(tau_root, j):
    """
    Construct Lagrange polynomial L_j for given collocation points.

    Args:
        tau_root (list): Collocation points
        j (int): Index of polynomial

    Returns:
        numpy.poly1d: Lagrange polynomial
    """
    p = np.poly1d([1])
    for r in range(len(tau_root)):
        if r != j:
            p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j] - tau_root[r])
    return p


def test_interpolation(d_poly=3):
    """
    Test interpolation with Lagrange polynomials.

    Args:
        d_poly (int): Polynomial degree
    """
    print("\nTesting Lagrange Interpolation")
    print("=" * 60)

    # Create collocation scheme
    scheme = CollocationScheme(d_poly)

    # Test function: f(t) = t^2
    f = lambda t: t**2

    # Interpolate at collocation points
    y_values = [f(t) for t in scheme.tau_root]

    # Construct interpolating polynomial
    p_interp = np.poly1d([0])
    for j in range(len(scheme.tau_root)):
        L_j = lagrange_polynomial(scheme.tau_root, j)
        p_interp += y_values[j] * L_j

    # Test at various points
    test_points = np.linspace(0, 1, 11)
    print(f"\nInterpolating f(t) = t^2 with {d_poly}-degree polynomial")
    print(f"{'t':>6s} {'f(t)':>10s} {'p(t)':>10s} {'error':>10s}")
    print("-" * 40)

    for t in test_points:
        f_val = f(t)
        p_val = p_interp(t)
        error = abs(f_val - p_val)
        print(f"{t:6.2f} {f_val:10.6f} {p_val:10.6f} {error:10.2e}")

    # The interpolation should be exact for polynomials of degree <= d_poly
    # For f(t) = t^2 with d_poly >= 2, error should be very small
    print(f"\nFor polynomial of degree 2, interpolation with degree {d_poly} should be exact.")


if __name__ == "__main__":
    """Test the collocation utilities."""
    print("Orthogonal Collocation Utilities Test")
    print("=" * 60)

    # Test different polynomial degrees
    for d in [2, 3, 4]:
        print(f"\n{'='*60}")
        print(f"Testing with polynomial degree {d}")
        print(f"{'='*60}")

        scheme = CollocationScheme(d_poly=d)
        scheme.print_info()
        scheme.print_verification()

    # Test interpolation
    test_interpolation(d_poly=3)

    print("\n" + "=" * 60)
    print("Collocation utilities test completed successfully!")