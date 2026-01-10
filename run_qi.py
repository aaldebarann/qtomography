from qiskit_quantuminspire.qi_provider import QIProvider

provider = QIProvider()
print(provider.backends())

# Get Quantum Inspire's simulator backend:
simulator_backend = provider.get_backend("QX emulator")

from qiskit import QuantumCircuit

# Create a basic Bell State circuit:
qc1 = QuantumCircuit(9, 9)
qc1.h(0)
qc1.cx(0, 1)
qc1.measure([0, 1], [0, 1])
qc2 = QuantumCircuit(9, 9)
qc2.h(0)
qc2.cx(0, 2)
qc2.measure([0, 2], [0, 2])

# Show a representation of the quantum circuit:
print(qc1, qc2)

# Run the circuit on Quantum Inspire's platform:
job = simulator_backend.run([qc1, qc2], shots=4096)

job.serialize("job.qpy")
# Print the results.
result = job.result()
print(result.get_counts())

# Get the error messages corresponding to each failed circuit
print(result.system_messages)