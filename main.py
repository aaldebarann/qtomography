import qiskit
import numpy as np
from utils import *

num_qubits = 2
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)
if num_qubits == 1:
    projectors, labels = naive_measure_operators(num_qubits)
    print(f"projectors:")
    for p in projectors:
        print(p)
    basis = qiskit.quantum_info.pauli_basis(1).to_matrix()[1:]
    basis = [b / norm(b) for b in basis]
    init_state = state_I
    freqs = measure(labels, init_state, num_qubits)
    print('freqs =\n', freqs)
    X = decompose_by_basis(basis, projectors)
    theta = solve_exact(X, freqs - 0.5)
    print(f'theta =\n{theta}')
    rho = np.eye(2 ** num_qubits) /2 + reconstruct_by_basis(basis, theta)
    print(f'X =\n{X}')
    print(f'reconstructed density matrix:\n{rho}')  
else:
    projectors, labels = naive_measure_operators(num_qubits)
    print(f"projectors:")
    for p in projectors:
        print(p)
    basis = qiskit.quantum_info.pauli_basis(num_qubits).to_matrix()[1:]
    basis = [b / norm(b) for b in basis]
    # print(f"basis = {basis}")
    init_state = state_I
    freqs = measure(labels, init_state, num_qubits)
    print('freqs =\n', freqs)
    X = decompose_by_basis(basis, projectors)
    print(f'X = \n{np.astype(X, float)*2}')
    theta = solve_exact(X, freqs - 0.5)
    print(f'theta =\n{theta}')
    dim = 2 ** num_qubits
    rho = np.eye(dim) / dim + reconstruct_by_basis(basis, theta)
    print(f'reconstructed density matrix:\n{rho}')