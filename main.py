import qiskit
import numpy as np
from utils import *
from qst import *
import scipy
import os
import scipy.linalg as la
from qiskit.quantum_info import Statevector, DensityMatrix, state_fidelity
from qiskit.circuit import QuantumCircuit
import math

num_qubits = int(os.environ.get('NUM_QUBITS', 1))
measure_mode = int(os.environ.get('MMODE', 0))
if measure_mode == 0:
    get_measure_operators = manual_measure_operators
    print("MANUAL")
elif measure_mode == 1:
    print("SILLY")
    get_measure_operators = silly_measure_operators
else:
    print("not_so SILLY")
    get_measure_operators = not_so_silly_measure_operators

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

init_state = state_I

measures = get_measure_operators(num_qubits)
print(f"projectors:")
# for i, m in enumerate(measures):
#     print(f'{i+1}:   {m}')
#     # print(get_povm_op(*m))
#     # print(p)

qc = QuantumCircuit(num_qubits)
qc = init_state(qc)
print("real density matrix")
statevector = Statevector(qc)
expected = DensityMatrix(statevector)
print(DensityMatrix(statevector))

basis = qiskit.quantum_info.pauli_basis(num_qubits).to_matrix()[1:]
basis = [b / norm(b) for b in basis]
# print(f"basis = {basis}")
freqs = measure(measures, init_state, num_qubits, 48000)
for i, m in enumerate(measures):
    print(f'{i+1}:   {m}  --> {round(freqs[i], 2)}')
    # print(get_povm_op(*m))
    # print(p)
# freqs = np.asarray([0.25, 0.25, 0.5, 0.25, 0.25, 0., 0.25, 0.5, 0., 1., 0.5, 0.25, 0.25, 0.5, 0.25,])
print('freqs =\n', freqs)
projectors = [get_povm_op(*label) for label in measures]
X = decompose_by_basis(basis, projectors)
print(f'X condition number: {np.linalg.cond(X)}')
print(f'X singular values: {np.linalg.svdvals(X)}')
# print(f'X = \n{np.astype(X, float)*2}')
dim = 2 ** num_qubits
theta = solve_exact(X, freqs - 1./dim)
print(f'theta =\n{theta}')
rho = np.eye(dim) / dim + reconstruct_by_basis(basis, theta)
print(f'reconstructed density matrix:\n{rho}')
print(f'Trace = {np.trace(rho)}')
eigenvalues = np.linalg.eigh(rho)[0]
positive_defined = np.all(eigenvalues > 0)
print(f'Positive-defined = {np.all(eigenvalues > 0)}')
print(f'Print eigenval = {eigenvalues}')

real = DensityMatrix(rho)
print(f'Fidelity: {state_fidelity(expected, expected)}')
print(f'Purity: {real.purity()}')
from matplotlib import pyplot as plt
diff = real - expected
print(f'Max diff: {np.max(np.abs(diff))}')
print(f'L2 norm: {np.sqrt(np.sum(np.square(np.abs(diff))))}')
# real.draw('city')
# plt.show()
# diff.draw('city')
# plt.show()
# X@theta = freqs