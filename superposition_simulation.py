import cirq
import qsimcirq
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def run_superposition_simulation():
    # 1. Specify one qubit. GridQubit is commonly used in Cirq.
    q = cirq.GridQubit(0, 0)

    # 2. Create a circuit.
    # Circuit after measurement (circuit_a)
    circuit_a = cirq.Circuit(
        # Apply the Hadamard gate (H) to put the qubit in superposition
        cirq.H(q),
        # Measure the qubit to observe the result (this collapses the superposition)
        cirq.measure(q, key='result'))

    # Circuit before measurement (circuit_b)
    circuit_b = cirq.Circuit(
        # Apply the Hadamard gate (H) to put the qubit in superposition
        cirq.H(q))

    # Print the circuit (optional)
    print("Quantum Circuit:")
    print(circuit_a)

    # 3. Run the circuit on the simulator.
    # We run it multiple times (e.g., 100 times) to see the probability distribution.
    simulator = qsimcirq.QSimSimulator()
    result = simulator.run(circuit_a, repetitions=1000)

    # 4. Print the measurement results.
    print("\nMeasurement results after 1000 repetitions:")
    counts = result.histogram(key='result')
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

    # Plotting measurement results
    # Convert Counter to lists for plotting
    measurement_outcomes = list(counts.keys())
    probabilities = list(counts.values())

    # Create the bar chart
    plt.figure(figsize=(6, 4))
    plt.bar(measurement_outcomes, probabilities, color=['skyblue', 'lightcoral'])

    # Add labels and title
    plt.xlabel('Measurement Outcome')
    plt.ylabel('Number of Occurrences')
    plt.title('Quantum Measurement Results')
    plt.xticks(measurement_outcomes) # Ensure ticks are at 0 and 1
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    print(counts)

    # Bloch Sphere Visualization
    alph = initial_state[0]
    beta = initial_state[1]

    # Calculate Bloch coordinates for the superposition state
    x_superposition = 2 * np.real(alph * np.conjugate(beta))
    y_superposition = 2 * np.imag(alph * np.conjugate(beta))
    z_superposition = np.abs(alph)**2 - np.abs(beta)**2

    # Create a new figure for the Bloch sphere
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Draw the Bloch sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the surface
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

    # Plot the qubit superposition state vector
    ax.quiver(0, 0, 0, x_superposition, y_superposition, z_superposition, color='r', linewidth=2, arrow_length_ratio=0.1, label='Superposition State')
    ax.scatter(x_superposition, y_superposition, z_superposition, color='r', s=100) # Plot the point on the sphere

    # --- Add dynamic visualization of measurement results ---
    count_0 = counts.get(0, 0)
    count_1 = counts.get(1, 0)
    total_repetitions = sum(counts.values())

    if total_repetitions > 0:
        percent_0 = (count_0 / total_repetitions) * 100
        percent_1 = (count_1 / total_repetitions) * 100
    else:
        percent_0 = 0
        percent_1 = 0

    # Plot the |0> state pole (North pole) as a marker
    ax.scatter(0, 0, 1, color='g', s=100, marker='o', label='|0> Measurement Outcome')
    ax.text(0.1, 0.1, 1.05, f'|0> ({count_0} - {percent_0:.1f}%)', color='g', fontsize=10)

    # Plot the |1> state pole (South pole) as a marker
    ax.scatter(0, 0, -1, color='orange', s=100, marker='o', label='|1> Measurement Outcome')
    ax.text(0.1, 0.1, -1.05, f'|1> ({count_1} - {percent_1:.1f}%)', color='orange', fontsize=10)

    ax.set_title('Bloch Sphere: Superposition State and Measurement Outcomes')
    ax.set_box_aspect([1,1,1]) # Equal aspect ratio
    ax.legend()

    plt.show()

if __name__ == '__main__':
    run_superposition_simulation()
