import cirq
import qsimcirq
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plot_bloch_sphere_measured_state(ax, qubit_name, counts_0, counts_1, total_repetitions):
    """Plots a Bloch sphere for a single qubit with measured state outcomes."""

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

    if total_repetitions > 0:
        percent_0 = (counts_0 / total_repetitions) * 100
        percent_1 = (counts_1 / total_repetitions) * 100
    else:
        percent_0 = 0
        percent_1 = 0

    # Plot the |0> state pole (North pole) as a marker
    ax.scatter(0, 0, 1, color='g', s=150, marker='o', label=f'|0> ({counts_0} - {percent_0:.1f}%)')
    ax.text(0.1, 0.1, 1.05, f'|0> ({percent_0:.1f}%)', color='g', fontsize=10)

    # Plot the |1> state pole (South pole) as a marker
    ax.scatter(0, 0, -1, color='orange', s=150, marker='o', label=f'|1> ({counts_1} - {percent_1:.1f}%)')
    ax.text(0.1, 0.1, -1.05, f'|1> ({percent_1:.1f}%)', color='orange', fontsize=10)

    ax.set_title(f'Bloch Sphere for Measured State of {qubit_name}')
    ax.set_box_aspect([1,1,1]) # Equal aspect ratio
    ax.legend(loc='lower left')

def run_bell_state_circuit():
    # 1. Define two qubits
    q0 = cirq.GridQubit(0, 0)
    q1 = cirq.GridQubit(0, 1)

    # 2. Create an empty cirq.Circuit object
    circuit_a = cirq.Circuit(
        # Apply a Hadamard gate (cirq.H) to the first qubit (q0)
        cirq.H(q0),
        # Apply a CNOT gate (cirq.CNOT) with q0 as the control and q1 as the target
        cirq.CNOT(q0, q1),
        # Add measurement operations to both qubits
        cirq.measure(q0, q1, key='final_result')
    )

    circuit_b = cirq.Circuit(
        # Apply a Hadamard gate (cirq.H) to the first qubit (q0)
        cirq.H(q0),
        # Apply a CNOT gate (cirq.CNOT) with q0 as the control and q1 as the target
        cirq.CNOT(q0, q1)
    )


    # 3. Print the created circuit to visualize its structure
    print("Bell State Circuit:")
    print(circuit_a)

    # 4. Run the circuit on the simulator.
    # We run it multiple times (e.g., 100 times) to see the probability distribution.
    simulator = qsimcirq.QSimSimulator()
    result = simulator.run(circuit_a, repetitions=1000)

    # 4. Print the measurement results.
    print("\nMeasurement results after 1000 repetitions:")
    counts = result.histogram(key='final_result')
    print(counts)

    # You can also use cirq.sample_state_vector to view the state vector directly
    # without measurement, which shows the complex superposition coefficients:
    print("\nVector status (before measurement):")
    initial_state = simulator.simulate(circuit_b).final_state_vector
    print(initial_state)

    # Vector after measurement
    print("\nVector status (after measurement):")
    after_m = simulator.simulate(circuit_a).final_state_vector
    print(after_m)

    # Retrieve counts from the Bell state simulation
    # The 'counts' variable stores measurement results as integers where 0='00', 1='01', 2='10', 3='11'

    # Initialize counts for each qubit
    q0_counts_0 = 0
    q0_counts_1 = 0
    q1_counts_0 = 0
    q1_counts_1 = 0

    total_repetitions = sum(counts.values())

    for outcome, num_occurrences in counts.items():
        # outcome is an integer, convert to binary string to check individual qubit states
        # For 2 qubits, outcome 0 is '00', 1 is '01', 2 is '10', 3 is '11'
        binary_outcome = format(outcome, '02b') # '02b' ensures 2-bit binary representation

        if binary_outcome[0] == '0': # First qubit (q0) is 0
            q0_counts_0 += num_occurrences
        else: # First qubit (q0) is 1
            q0_counts_1 += num_occurrences

        if binary_outcome[1] == '0': # Second qubit (q1) is 0
            q1_counts_0 += num_occurrences
        else: # Second qubit (q1) is 1
            q1_counts_1 += num_occurrences

    # Create a figure with two subplots
    fig = plt.figure(figsize=(16, 8))

    # Plot for Qubit 0
    ax0 = fig.add_subplot(1, 2, 1, projection='3d')
    plot_bloch_sphere_measured_state(ax0, 'Qubit 0', q0_counts_0, q0_counts_1, total_repetitions)

    # Plot for Qubit 1
    ax1 = fig.add_subplot(1, 2, 2, projection='3d')
    plot_bloch_sphere_measured_state(ax1, 'Qubit 1', q1_counts_0, q1_counts_1, total_repetitions)

    plt.tight_layout()
    plt.show()

    # The final_state_vector is available from the simulation for circuit_b
    # (before measurement)
    # initial_state = simulator.simulate(circuit_b).final_state_vector

    # Calculate probabilities from the amplitudes (square of the absolute values)
    probabilities = np.abs(initial_state)

    # Define labels for the computational basis states
    # For two qubits, the states are |00>, |01>, |10>, |11>
    state_labels = ['|00>', '|01>', '|10>', '|11>']

    # Create the bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(state_labels, probabilities, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])

    # Add labels and title
    plt.xlabel('Computational Basis State')
    plt.ylabel('Probability')
    plt.title('Probabilities of Bell State Computational Basis (Before Measurement)')
    plt.ylim(0, 1) # Probabilities range from 0 to 1
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

if __name__ == '__main__':
    run_bell_state_circuit()
