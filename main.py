import qiskit
import numpy as np
from utils import *
import scipy
import os

num_qubits = int(os.environ.get('NUM_QUBITS', 1))
measure_mode = int(os.environ.get('MMODE', 0))
if measure_mode == 0:
    get_measure_operators = manual_measure_operators
elif measure_mode == 1:
    get_measure_operators = silly_measure_operators
else:
    get_measure_operators = not_so_silly_measure_operators

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

init_state = state_I


if num_qubits == 1:
    projectors, labels = get_measure_operators(num_qubits)
    print(f"projectors:")
    for p in projectors:
        print(p)
    basis = qiskit.quantum_info.pauli_basis(1).to_matrix()[1:]
    basis = [b / norm(b) for b in basis]
    freqs = measure(labels, init_state, num_qubits)
    print('freqs =\n', freqs)
    X = decompose_by_basis(basis, projectors)
    print(f'X =\n{X}')
    theta = solve_exact(X, freqs - 0.5)
    print(f'theta =\n{theta}')
    rho = np.eye(2 ** num_qubits) /2 + reconstruct_by_basis(basis, theta)
    print(f'reconstructed density matrix:\n{rho}')  
else:
    projectors, labels = get_measure_operators(num_qubits)
    print(f"projectors:")
    for l, p in zip(labels, projectors):
        print(l)
        # print(p)
    qc = QuantumCircuit(num_qubits)
    qc = init_state(qc)
    from qiskit.quantum_info import Statevector, DensityMatrix
    from qiskit.circuit import QuantumCircuit
    print("read density matrix")
    statevector = Statevector(qc)
    print(DensityMatrix(statevector))
    basis = qiskit.quantum_info.pauli_basis(num_qubits).to_matrix()[1:]
    basis = [b / norm(b) for b in basis]
    # print(f"basis = {basis}")
    freqs = measure(labels, init_state, num_qubits, 48000)
    # print('freqs =\n', freqs)
    X = decompose_by_basis(basis, projectors)
    # print(f'X = \n{np.astype(X, float)*2}')
    dim = 2 ** num_qubits
    theta = solve_exact(X, freqs - 1./dim)
    # print(f'theta =\n{theta}')
    rho = np.eye(dim) / dim + reconstruct_by_basis(basis, theta)
    print(f'reconstructed density matrix:\n{rho}')

    # expected = np.zeros(16)
    # expected[0] = 1
    # i = np.eye(dim).flatten() / dim
    # a = []
    # for b in basis:
    #     a.append(b.flatten())
    # a.append(i)
    # a = np.asarray(a).T
    # expected_theta = scipy.linalg.solve(a, expected)
    # print(f"expected theta = {expected_theta}")
    # rho = np.eye(dim) / dim + reconstruct_by_basis(basis, expected_theta)
    # print(f'reconstructed density matrix:\n{rho}')

# X@theta = freqs