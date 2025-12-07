"""
Visualization Utilities
========================

This module provides plotting functions for visualizing
MPC results, including state trajectories, control inputs,
and power systems.

Author: Advanced Process Control Project
Course: WS 2025/2026
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from swimmer_model import get_default_parameters


# Set default matplotlib parameters
mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.figsize'] = (14, 10)
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.3


def plot_state_trajectories(X_opt, dt, title="State Trajectories", save_path=None):
    """
    Plot all state trajectories.

    Args:
        X_opt (array): State trajectory [nx, N+1]
        dt (float): Time step
        title (str): Plot title
        save_path (str): Path to save figure (optional)
    """
    time_vec = np.arange(X_opt.shape[1]) * dt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Distance
    axes[0, 0].plot(time_vec, X_opt[0, :], 'b-', linewidth=2)
    axes[0, 0].axhline(y=100, color='r', linestyle='--', linewidth=1, label='Target (100m)')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Distance (m)')
    axes[0, 0].set_title('Distance Traveled')
    axes[0, 0].legend()

    # Velocity
    axes[0, 1].plot(time_vec, X_opt[1, :], 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Velocity (m/s)')
    axes[0, 1].set_title('Velocity Profile')

    # Alactacid Energy
    params = get_default_parameters()
    axes[1, 0].plot(time_vec, X_opt[2, :], 'g-', linewidth=2)
    axes[1, 0].axhline(y=params['Emax_al'], color='g', linestyle='--', 
                       linewidth=1, alpha=0.5, label='Maximum')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Alactacid Energy (J)')
    axes[1, 0].set_title('Remaining Alactacid Energy')
    axes[1, 0].legend()

    # Fatigue
    axes[1, 1].plot(time_vec, X_opt[3, :], 'm-', linewidth=2)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Fatigue (mmol/l)')
    axes[1, 1].set_title('Fatigue Level')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_control_inputs(U_opt, dt, title="Control Inputs", save_path=None):
    """
    Plot control input trajectories.

    Args:
        U_opt (array): Control trajectory [nu, N]
        dt (float): Time step
        title (str): Plot title
        save_path (str): Path to save figure (optional)
    """
    time_vec = np.arange(U_opt.shape[1]) * dt
    params = get_default_parameters()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Propulsion Force
    axes[0].step(time_vec, U_opt[0, :], 'b-', linewidth=2, where='post', label='Force')
    axes[0].axhline(y=params['umax'], color='r', linestyle='--', 
                    linewidth=1, label='Maximum')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Force (N)')
    axes[0].set_title('Propulsion Force')
    axes[0].legend()

    # Anaerobic Power
    axes[1].step(time_vec, U_opt[1, :], 'r-', linewidth=2, where='post')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Anaerobic Power (W)')
    axes[1].set_title('Anaerobic Power Usage')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_power_systems(powers, dt, title="Power Systems", save_path=None):
    """
    Plot all power system trajectories.

    Args:
        powers (dict): Dictionary of power arrays
        dt (float): Time step
        title (str): Plot title
        save_path (str): Path to save figure (optional)
    """
    N = len(powers['P_mech'])
    time_vec = np.arange(N) * dt
    params = get_default_parameters()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # All powers together
    axes[0, 0].plot(time_vec, powers['P_body'], 'k-', linewidth=2, label='Body')
    axes[0, 0].plot(time_vec, powers['P_al'], 'g-', linewidth=2, label='Alactacid')
    axes[0, 0].plot(time_vec, powers['P_anaerobic'], 'r-', linewidth=2, label='Anaerobic')
    axes[0, 0].plot(time_vec, powers['P_aerob'], 'b-', linewidth=2, label='Aerobic')
    axes[0, 0].axhline(y=params['Pmax_aerobic'], color='b', linestyle='--', 
                       linewidth=1, alpha=0.5)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Power (W)')
    axes[0, 0].set_title('All Power Systems')
    axes[0, 0].legend()

    # Mechanical Power
    axes[0, 1].plot(time_vec, powers['P_mech'], 'purple', linewidth=2)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Mechanical Power (W)')
    axes[0, 1].set_title('Mechanical Power Output')

    # Energy System Breakdown
    axes[1, 0].fill_between(time_vec, 0, powers['P_al'], 
                            alpha=0.5, color='g', label='Alactacid')
    axes[1, 0].fill_between(time_vec, powers['P_al'], 
                            powers['P_al'] + powers['P_anaerobic'],
                            alpha=0.5, color='r', label='Anaerobic')
    axes[1, 0].fill_between(time_vec, 
                            powers['P_al'] + powers['P_anaerobic'],
                            powers['P_body'],
                            alpha=0.5, color='b', label='Aerobic')
    axes[1, 0].plot(time_vec, powers['P_body'], 'k-', linewidth=2, label='Total')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Power (W)')
    axes[1, 0].set_title('Energy System Breakdown (Stacked)')
    axes[1, 0].legend()

    # Power efficiency
    efficiency = powers['P_mech'] / (powers['P_body'] + 1e-6)  # Avoid division by zero
    axes[1, 1].plot(time_vec, efficiency * 100, 'orange', linewidth=2)
    axes[1, 1].axhline(y=params['beta'] * 100, color='r', linestyle='--', 
                       linewidth=1, label=f"Design: {params['beta']*100:.0f}%")
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Efficiency (%)')
    axes[1, 1].set_title('Power Efficiency')
    axes[1, 1].legend()
    axes[1, 1].set_ylim([0, 100])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_complete_results(X_opt, U_opt, powers, dt, title="Complete Results", 
                         save_path=None):
    """
    Create comprehensive plot with all results.

    Args:
        X_opt (array): State trajectory
        U_opt (array): Control trajectory
        powers (dict): Power trajectories
        dt (float): Time step
        title (str): Plot title
        save_path (str): Path to save figure (optional)
    """
    params = get_default_parameters()
    time_state = np.arange(X_opt.shape[1]) * dt
    time_control = np.arange(U_opt.shape[1]) * dt

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(title, fontsize=18, fontweight='bold')

    # Distance
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(time_state, X_opt[0, :], 'b-', linewidth=2)
    ax1.axhline(y=100, color='r', linestyle='--', label='Target')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Distance (m)')
    ax1.set_title('Distance')
    ax1.legend()

    # Velocity
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(time_state, X_opt[1, :], 'r-', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Velocity')

    # Alactacid Energy
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(time_state, X_opt[2, :], 'g-', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Energy (J)')
    ax3.set_title('Alactacid Energy')

    # Fatigue
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(time_state, X_opt[3, :], 'm-', linewidth=2)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Fatigue (mmol/l)')
    ax4.set_title('Fatigue')

    # Propulsion Force
    ax5 = plt.subplot(3, 3, 5)
    ax5.step(time_control, U_opt[0, :], 'b-', linewidth=2, where='post')
    ax5.axhline(y=params['umax'], color='r', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Force (N)')
    ax5.set_title('Propulsion Force')

    # Powers
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(time_control, powers['P_body'], 'k-', linewidth=2, label='Body')
    ax6.plot(time_control, powers['P_al'], 'g-', linewidth=1.5, label='Alactacid')
    ax6.plot(time_control, powers['P_anaerobic'], 'r-', linewidth=1.5, label='Anaerobic')
    ax6.plot(time_control, powers['P_aerob'], 'b-', linewidth=1.5, label='Aerobic')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Power (W)')
    ax6.set_title('Power Systems')
    ax6.legend(fontsize=8)

    # Mechanical Power
    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(time_control, powers['P_mech'], 'purple', linewidth=2)
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Power (W)')
    ax7.set_title('Mechanical Power')

    # Anaerobic Power
    ax8 = plt.subplot(3, 3, 8)
    ax8.step(time_control, U_opt[1, :], 'r-', linewidth=2, where='post')
    ax8.set_xlabel('Time (s)')
    ax8.set_ylabel('Power (W)')
    ax8.set_title('Anaerobic Power')

    # Force with Fatigue
    ax9 = plt.subplot(3, 3, 9)
    force_with_fatigue = U_opt[0, :] + params['alpha'] * X_opt[3, :-1]
    ax9.plot(time_control, force_with_fatigue, 'orange', linewidth=2, label='u + αF')
    ax9.axhline(y=params['umax'], color='r', linestyle='--', label='Max')
    ax9.set_xlabel('Time (s)')
    ax9.set_ylabel('Force (N)')
    ax9.set_title('Force Constraint')
    ax9.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_horizon_analysis(results, save_path=None):
    """
    Plot analysis of different prediction horizons.

    Args:
        results (list): List of result dictionaries with 'N', 'time', etc.
        save_path (str): Path to save figure (optional)
    """
    # Filter successful results
    successful = [r for r in results if r.get('success', False)]

    if not successful:
        print("No successful results to plot!")
        return

    N_vals = [r['N'] for r in successful]
    times = [r['time'] for r in successful]
    avg_vels = [r['avg_vel'] for r in successful]
    fatigues = [r['final_fatigue'] for r in successful]
    energies = [r['final_energy'] for r in successful]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Prediction Horizon Analysis', fontsize=16, fontweight='bold')

    # Race time
    axes[0, 0].plot(N_vals, times, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Prediction Horizon')
    axes[0, 0].set_ylabel('Race Time (s)')
    axes[0, 0].set_title('Race Time vs Horizon')

    # Average velocity
    axes[0, 1].plot(N_vals, avg_vels, 'ro-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Prediction Horizon')
    axes[0, 1].set_ylabel('Average Velocity (m/s)')
    axes[0, 1].set_title('Average Velocity vs Horizon')

    # Final fatigue
    axes[1, 0].plot(N_vals, fatigues, 'mo-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Prediction Horizon')
    axes[1, 0].set_ylabel('Final Fatigue (mmol/l)')
    axes[1, 0].set_title('Final Fatigue vs Horizon')

    # Final energy
    axes[1, 1].plot(N_vals, energies, 'go-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Prediction Horizon')
    axes[1, 1].set_ylabel('Final Alactacid Energy (J)')
    axes[1, 1].set_title('Final Energy vs Horizon')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    """Test visualization functions with synthetic data."""
    print("Testing Visualization Utilities")
    print("="*60)

    # Create synthetic data
    N = 50
    dt = 0.5

    # States
    X_opt = np.zeros((4, N+1))
    X_opt[0, :] = np.linspace(0, 100, N+1)  # distance
    X_opt[1, :] = 2.0 + 0.5 * np.sin(np.linspace(0, 2*np.pi, N+1))  # velocity
    X_opt[2, :] = 1000 * np.exp(-np.linspace(0, 3, N+1))  # energy decay
    X_opt[3, :] = 5 * (1 - np.exp(-np.linspace(0, 2, N+1)))  # fatigue growth

    # Controls
    U_opt = np.zeros((2, N))
    U_opt[0, :] = 100 + 50 * np.sin(np.linspace(0, 2*np.pi, N))  # force
    U_opt[1, :] = 150 + 50 * np.cos(np.linspace(0, 2*np.pi, N))  # anaerobic

    # Powers (synthetic)
    powers = {
        'P_mech': 200 + 50 * np.sin(np.linspace(0, 2*np.pi, N)),
        'P_body': 1000 + 200 * np.sin(np.linspace(0, 2*np.pi, N)),
        'P_al': 500 * np.exp(-np.linspace(0, 3, N)),
        'P_aerob': 300 * np.ones(N),
        'P_anaerobic': U_opt[1, :]
    }

    print("\nGenerating test plots...")

    # Test individual plot functions
    plot_state_trajectories(X_opt, dt, "Test State Trajectories")
    plot_control_inputs(U_opt, dt, "Test Control Inputs")
    plot_power_systems(powers, dt, "Test Power Systems")
    plot_complete_results(X_opt, U_opt, powers, dt, "Test Complete Results")

    print("\nVisualization test completed!")
