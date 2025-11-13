import cirq
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import qsimcirq

def qft_circuit(qubits):
  q_list = list(qubits)
  circuit = cirq.Circuit()
  # Step 1: Apply H gates and controlled rotations
  for i, qubit in enumerate(q_list):
    # Apply H to the i-th qubit
    circuit.append(cirq.H(qubit))

    # Apply controlled rotation to the next qubit
    for j in range(i + 1, len(q_list)):
      # Rotation angle depends on the distance between the control and target qubits
      angle = 2.0 * np.pi / (2**(j - i + 1))
      # Implement controlled rotation gate
      circuit.append(cirq.CZ(q_list[j], qubit)**(2 * angle / np.pi))

  # Perform SWAP to reverse the order of qubits
  for i in range(len(q_list) // 2):
    circuit.append(cirq.SWAP(q_list[i], q_list[len(q_list) - 1 - i]))

  return circuit

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

def run_quantum_fourier_transform():
    # Define 3 qubits
    q0, q1, q2 = cirq.GridQubit.rect(1, 3)

    # Create a Quantum Fourier Transform (QFT) circuit function
    my_qft = qft_circuit([q0, q1, q2])
    print("Quantum Fourier Transform (QFT) circuit for 3 qubits:")
    print(my_qft)

    # Input Circuit
    input_circuit = cirq.Circuit(
        cirq.X(q0),
        cirq.X(q2))

    full_circuit_qft_only = input_circuit + my_qft
    print("\nThis is the complete circuit for QFT only: ")
    print(full_circuit_qft_only)

    simulator = cirq.Simulator()
    result_qft_only = simulator.simulate(full_circuit_qft_only)
    print("\nState vector after QFT: ")
    print(np.round(result_qft_only.final_state_vector, 3))

    # Inverse QFT
    iqft_circuit = cirq.inverse(my_qft)

    full_circuit_qft_iqft = input_circuit + my_qft + iqft_circuit
    print("\nThis is the complete circuit with QFT and iQFT: ")
    print(full_circuit_qft_iqft)

    simulator = qsimcirq.QSimSimulator()
    result_qft_iqft = simulator.simulate(full_circuit_qft_iqft)
    print("\nState vector after QFT and iQFT: ")
    print(np.round(result_qft_iqft.final_state_vector, 3))

    # The 'result.final_state_vector' from the previous cell is the state we want to visualize
    final_qft_state_vector = result_qft_iqft.final_state_vector

    # Define system shape: (qubit 0, qubit 1, qubit 2) -> (dim 2, dim 2, dim 2)
    QID_SHAPE_3_QUBITS = (2, 2, 2)

    # Pauli matrices for Bloch vector calculation
    PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    PAULI_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    # --- Calculate Bloch vectors for each qubit ---

    # Qubit 0
    rho_q0_qft = cirq.density_matrix_from_state_vector(
        final_qft_state_vector, indices=[0], qid_shape=QID_SHAPE_3_QUBITS
    )
    x_q0_qft = np.real(np.trace(np.dot(rho_q0_qft, PAULI_X)))
    y_q0_qft = np.real(np.trace(np.dot(rho_q0_qft, PAULI_Y)))
    z_q0_qft = np.real(np.trace(np.dot(rho_q0_qft, PAULI_Z)))

    # Qubit 1
    rho_q1_qft = cirq.density_matrix_from_state_vector(
        final_qft_state_vector, indices=[1], qid_shape=QID_SHAPE_3_QUBITS
    )
    x_q1_qft = np.real(np.trace(np.dot(rho_q1_qft, PAULI_X)))
    y_q1_qft = np.real(np.trace(np.dot(rho_q1_qft, PAULI_Y)))
    z_q1_qft = np.real(np.trace(np.dot(rho_q1_qft, PAULI_Z)))

    # Qubit 2
    rho_q2_qft = cirq.density_matrix_from_state_vector(
        final_qft_state_vector, indices=[2], qid_shape=QID_SHAPE_3_QUBITS
    )
    x_q2_qft = np.real(np.trace(np.dot(rho_q2_qft, PAULI_X)))
    y_q2_qft = np.real(np.trace(np.dot(rho_q2_qft, PAULI_Y)))
    z_q2_qft = np.real(np.trace(np.dot(rho_q2_qft, PAULI_Z)))

    # --- Plotting Execution ---
    fig = plt.figure(figsize=(18, 6))

    ax0 = fig.add_subplot(1, 3, 1, projection='3d')
    plot_bloch_sphere(ax0, x_q0_qft, y_q0_qft, z_q0_qft, 'Bloch Sphere for Qubit 0 (After QFT+iQFT)')

    ax1 = fig.add_subplot(1, 3, 2, projection='3d')
    plot_bloch_sphere(ax1, x_q1_qft, y_q1_qft, z_q1_qft, 'Bloch Sphere for Qubit 1 (After QFT+iQFT)')

    ax2 = fig.add_subplot(1, 3, 3, projection='3d')
    plot_bloch_sphere(ax2, x_q2_qft, y_q2_qft, z_q2_qft, 'Bloch Sphere for Qubit 2 (After QFT+iQFT)')

    plt.tight_layout()
    plt.show()

    print(f"Bloch vector for q0: ({x_q0_qft:.3f}, {y_q0_qft:.3f}, {z_q0_qft:.3f})")
    print(f"Bloch vector for q1: ({x_q1_qft:.3f}, {y_q1_qft:.3f}, {z_q1_qft:.3f})")
    print(f"Bloch vector for q2: ({x_q2_qft:.3f}, {y_q2_qft:.3f}, {z_q2_qft:.3f})")

if __name__ == '__main__':
    run_quantum_fourier_transform()
