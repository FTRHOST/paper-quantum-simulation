import cirq
import qsimcirq
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_bloch_sphere(ax, x, y, z, title):
    """Plots a Bloch sphere with a given state vector."""
    # Draw the Bloch sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(xs, ys, zs, color='b', alpha=0.1, linewidth=0)

    # Plot the axes
    ax.plot([-1, 1], [0, 0], [0, 0], color='gray', linestyle='--') # X-axis
    ax.plot([0, 0], [-1, 1], [0, 0], color='gray', linestyle='--') # Y-axis
    ax.plot([0, 0], [0, 0], [-1, 1], color='gray', linestyle='--') # Z-axis

    # Labels for axes
    ax.text(1.1, 0, 0, '|+X>', color='black')
    ax.text(-1.1, 0, 0, '|-X>', color='black')
    ax.text(0, 1.1, 0, '|+Y>', color='black')
    ax.text(0, -1.1, 0, '|-Y>', color='black')
    ax.text(0, 0, 1.1, '|0>', color='black')
    ax.text(0, 0, -1.1, '|1>', color='black')

    # Plot the qubit state vector
    ax.quiver(0, 0, 0, x, y, z, color='r', linewidth=2, arrow_length_ratio=0.1, label='Qubit State')
    ax.scatter(x, y, z, color='r', s=100) # Plot the point on the sphere

    ax.set_title(title)
    ax.set_box_aspect([1,1,1]) # Equal aspect ratio
    ax.view_init(elev=20, azim=30) # Set viewing angle

def run_deutsch_josza_balanced_function():
    # Create two qubits
    q0 = cirq.GridQubit(0, 0)
    q1 = cirq.GridQubit(0, 1)

    # Create the circuit (Balanced Function)
    circuit_dj = cirq.Circuit(
        cirq.H(q0),
        cirq.X(q1),
        cirq.H(q1),
        cirq.CNOT(q0, q1), # Oracle for f(x)=x (balanced)
        cirq.H(q0),
        cirq.measure(q0, key='result'),
    )

    # Create the circuit (without measurement) for state vector visualization
    circuit_dj_a = cirq.Circuit(
        cirq.H(q0),
        cirq.X(q1),
        cirq.H(q1),
        cirq.CNOT(q0, q1),
        cirq.H(q0),
    )

    print("Circuit for Balanced Function: ")
    print(circuit_dj)

    # Create simulator
    simulator = qsimcirq.QSimSimulator()
    result = simulator.run(circuit_dj, repetitions=1000)

    print("Simulation results: ")
    print(result.histogram(key='result'))

    # Get the state vector before measurement for Bloch sphere visualization
    print("\nVector status (before measurement):")
    initial_state = simulator.simulate(circuit_dj_a).final_state_vector
    print(initial_state)

    # Vector after measurement
    print("\nVector status (after measurement):")
    after_m = simulator.simulate(circuit_dj).final_state_vector
    print(after_m)

    # Define system shape: (qubit 0, qubit 1) -> (dim 2, dim 2)
    QID_SHAPE = (2, 2)

    # Pauli matrices for Bloch vector calculation
    PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    PAULI_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    # Calculate reduced density matrix for q0 (index 0)
    rho_q0 = cirq.density_matrix_from_state_vector(
        initial_state, indices=[0], qid_shape=QID_SHAPE
    )

    # Calculate Bloch vector for q0
    x_q0 = np.real(np.trace(np.dot(rho_q0, PAULI_X)))
    y_q0 = np.real(np.trace(np.dot(rho_q0, PAULI_Y)))
    z_q0 = np.real(np.trace(np.dot(rho_q0, PAULI_Z)))

    # Calculate reduced density matrix for q1 (index 1)
    rho_q1 = cirq.density_matrix_from_state_vector(
        initial_state, indices=[1], qid_shape=QID_SHAPE
    )

    # Calculate Bloch vector for q1
    x_q1 = np.real(np.trace(np.dot(rho_q1, PAULI_X)))
    y_q1 = np.real(np.trace(np.dot(rho_q1, PAULI_Y)))
    z_q1 = np.real(np.trace(np.dot(rho_q1, PAULI_Z)))

    # Create a figure with two subplots
    fig = plt.figure(figsize=(16, 8))

    # Plot for Qubit 0
    ax0 = fig.add_subplot(1, 2, 1, projection='3d')
    plot_bloch_sphere(ax0, x_q0, y_q0, z_q0, 'Bloch Sphere for Qubit 0 (Reduced State)')

    # Plot for Qubit 1
    ax1 = fig.add_subplot(1, 2, 2, projection='3d')
    plot_bloch_sphere(ax1, x_q1, y_q1, z_q1, 'Bloch Sphere for Qubit 1 (Reduced State)')

    plt.tight_layout()
    plt.show()

    print(f"Bloch vector for q0: ({x_q0:.3f}, {y_q0:.3f}, {z_q0:.3f})")
    print(f"Bloch vector for q1: ({x_q1:.3f}, {y_q1:.3f}, {z_q1:.3f})")

if __name__ == '__main__':
    run_deutsch_josza_balanced_function()
