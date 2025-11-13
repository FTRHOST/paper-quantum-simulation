import cirq
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import qsimcirq

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
    ax.view_init(elev=20, azim=30) # Atur sudut pandang

def run_deutsch_josza_constant_function():
    # Import cirq and qsimcirq
    

    # Create two qubits
    q0 = cirq.GridQubit(0, 0)
    q1 = cirq.GridQubit(0, 1)

    # Create the circuit (Constant Function)
    # For a constant function f(x)=0, the oracle is an identity operation
    # For f(x)=1, it's equivalent to applying X on q1 before and after the 'oracle' block.
    circuit_dj_c = cirq.Circuit(
        cirq.H(q0),
        cirq.X(q1),
        cirq.H(q1),
        # The 'oracle' for f(x)=0 is effectively no operation (identity) here.
        # For f(x)=1, you would apply cirq.X(q1) here as well.
        cirq.H(q0),
        cirq.measure(q0, key='result'),
    )

    # Create the circuit without measurement to get the final state vector
    circuit_dj_c_no_measure = cirq.Circuit(
        cirq.H(q0),
        cirq.X(q1),
        cirq.H(q1),
        cirq.H(q0)
    )

    print("Circuit for Constant Function:")
    print(circuit_dj_c)

    # Create simulator
    simulator = qsimcirq.QSimSimulator()
    result = simulator.run(circuit_dj_c, repetitions=100)

    print("\nSimulation results (for q0):")
    print(result.histogram(key='result'))

    # You can also use cirq.sample_state_vector to view the state vector directly
    # without measurement, which shows the complex superposition coefficients:
    print("\nVector status (before measurement):")
    initial_state_constant_function = simulator.simulate(circuit_dj_c_no_measure).final_state_vector
    print(initial_state_constant_function)

    # Vector after measurement
    print("\nVector status (after measurement):")
    after_m_constant = simulator.simulate(circuit_dj_c).final_state_vector
    print(after_m_constant)


    # Tentukan bentuk sistem: (qubit 0, qubit 1) -> (dim 2, dim 2)
    QID_SHAPE = (2, 2)

    # Pauli matrices for Bloch vector calculation
    PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    PAULI_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    # Hitung reduced density matrix untuk q0 (indeks 0)
    rho_q0_const = cirq.density_matrix_from_state_vector(
        initial_state_constant_function, indices=[0], qid_shape=QID_SHAPE
    )

    # Hitung Bloch vector for q0
    x_q0_const = np.real(np.trace(np.dot(rho_q0_const, PAULI_X)))
    y_q0_const = np.real(np.trace(np.dot(rho_q0_const, PAULI_Y)))
    z_q0_const = np.real(np.trace(np.dot(rho_q0_const, PAULI_Z)))

    # Hitung reduced density matrix untuk q1 (indeks 1)
    rho_q1_const = cirq.density_matrix_from_state_vector(
        initial_state_constant_function, indices=[1], qid_shape=QID_SHAPE
    )

    # Hitung Bloch vector for q1
    x_q1_const = np.real(np.trace(np.dot(rho_q1_const, PAULI_X)))
    y_q1_const = np.real(np.trace(np.dot(rho_q1_const, PAULI_Y)))
    z_q1_const = np.real(np.trace(np.dot(rho_q1_const, PAULI_Z)))

    # --- Eksekusi Plotting ---

    # Create a figure with two subplots
    fig = plt.figure(figsize=(16, 8))

    # Plot for Qubit 0
    ax0 = fig.add_subplot(1, 2, 1, projection='3d')
    plot_bloch_sphere(ax0, x_q0_const, y_q0_const, z_q0_const, 'Bloch Sphere for Qubit 0 (Constant Function)')

    # Plot for Qubit 1
    ax1 = fig.add_subplot(1, 2, 2, projection='3d')
    plot_bloch_sphere(ax1, x_q1_const, y_q1_const, z_q1_const, 'Bloch Sphere for Qubit 1 (Constant Function)')

    plt.tight_layout()
    plt.show()

    print(f"Bloch vector for q0 (constant function): ({x_q0_const:.3f}, {y_q0_const:.3f}, {z_q0_const:.3f})")
    print(f"Bloch vector for q1 (constant function): ({x_q1_const:.3f}, {y_q1_const:.3f}, {z_q1_const:.3f})")

if __name__ == '__main__':
    run_deutsch_josza_constant_function()
