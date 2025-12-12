import numpy as np
import qiskit
import qiskit.quantum_info

print(qiskit.quantum_info.pauli_basis(2).to_labels()[1:])