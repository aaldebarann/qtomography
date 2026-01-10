import qiskit
import numpy as np
from utils import *
from qst import *
import scipy
import os
import scipy.linalg as la
from qiskit.quantum_info import Statevector, DensityMatrix, state_fidelity, partial_trace
from qiskit.circuit import QuantumCircuit
import math
import argparse
import datetime

# parser = argparse.ArgumentParser(
#          prog='ProgramName',
#          description='What the program does',
#          epilog='Text at the bottom of help')

# parser.add_argument('-b', '--backend',
#                     choices=['statevector', 'qx', 'tuna9'],
#                     required=True,
#                     help='bachend type')
# parser.add_argument('--max',
#                     type=int,
#                     required=True,
#                     help='max qubits')
random_numbers = [0.7170362821184543, 0.16111541323504674, 0.2090945149664063]
# import matplotlib.pyplot as plt


# qmap = [4, 1, 2, 6, 7] # mapping on Tuna-9 qubits
qmap = [4, 1, 2] # mapping on Tuna-9 qubits


def get_test_1_qubit(idx):
    def init(circ):
        circ.h(idx)
        circ.rx(random_numbers[0], idx)
        circ.ry(random_numbers[1], idx)
        circ.rz(random_numbers[2], idx)
        return circ
    return init

def get_test_2_qubit(center, idx):
    def init(circ):
        circ.h(center)
        circ.h(idx)
        circ.cz(center, idx)
        circ.h(idx)
        circ.rx(random_numbers[0], idx)
        circ.ry(random_numbers[1], idx)
        circ.rz(random_numbers[2], idx)
        return circ
    return init

def get_test_3_qubit(center, i1, i2):
    def init(circ):
        idx1 = qmap[i1]
        idx2 = qmap[i2]
        circ.h(center)
        circ.h(idx1)
        circ.h(idx2)
        circ.cz(center, idx1)
        circ.cz(center, idx2)
        circ.h(idx1)
        circ.h(idx2)
        circ.rx(random_numbers[0], idx1)
        circ.ry(random_numbers[1], idx1)
        circ.rz(random_numbers[2], idx1)
        circ.rx(random_numbers[0], idx2)
        circ.ry(random_numbers[1], idx2)
        circ.rz(random_numbers[2], idx2)
        return circ
    return init

def get_test_4_qubit(center, i1, i2, i3):
    def init(circ):
        idx1 = qmap[i1]
        idx2 = qmap[i2]
        idx3 = qmap[i3]
        
        circ = QuantumCircuit(9, 9)
        circ.h(center)
        circ.h(idx1)
        circ.h(idx2)
        circ.h(idx3)
        circ.cz(center, idx1)
        circ.cz(center, idx2)
        circ.cz(center, idx3)
        circ.h(idx1)
        circ.h(idx2)
        circ.h(idx2)
        circ.rx(random_numbers[0], idx1)
        circ.ry(random_numbers[1], idx1)
        circ.rz(random_numbers[2], idx1)
        circ.rx(random_numbers[0], idx2)
        circ.ry(random_numbers[1], idx2)
        circ.rz(random_numbers[2], idx2)
        circ.rx(random_numbers[0], idx3)
        circ.ry(random_numbers[1], idx3)
        circ.rz(random_numbers[2], idx3)
        return circ
    return init

def get_test_5_qubit(center, i1, i2, i3, i4):
    def init(circ):
        idx1 = qmap[i1]
        idx2 = qmap[i2]
        idx3 = qmap[i3]
        idx4 = qmap[i4]
        
        circ = QuantumCircuit(9, 9)
        circ.h(center)
        circ.h(idx1)
        circ.h(idx2)
        circ.h(idx3)
        circ.h(idx4)
        circ.cz(center, idx1)
        circ.cz(center, idx2)
        circ.cz(center, idx3)
        circ.cz(center, idx4)
        circ.h(idx1)
        circ.h(idx2)
        circ.h(idx2)
        circ.h(idx4)
        circ.rx(random_numbers[0], idx1)
        circ.ry(random_numbers[1], idx1)
        circ.rz(random_numbers[2], idx1)
        circ.rx(random_numbers[0], idx2)
        circ.ry(random_numbers[1], idx2)
        circ.rz(random_numbers[2], idx2)
        circ.rx(random_numbers[0], idx3)
        circ.ry(random_numbers[1], idx3)
        circ.rz(random_numbers[2], idx3)
        circ.rx(random_numbers[0], idx4)
        circ.ry(random_numbers[1], idx4)
        circ.rz(random_numbers[2], idx4)
        return circ
    return init

def get_init_states():
    res = []
    center = qmap[0]
    # one-qubit circuits
    for idx in qmap:
        res.append(get_test_1_qubit(idx))
    # two-qubit circuits
    for idx in qmap[1:]:
        res.append(get_test_2_qubit(center, idx))
    
    # three-qubit circuits
    for i1 in range(1, len(qmap)):
        for i2 in range(i1 + 1, len(qmap)):
            res.append(get_test_3_qubit(center, i1, i2))
    
    # # four-qubit circuits
    # for i1 in range(1, len(qmap)):
    #     for i2 in range(i1 + 1, len(qmap)):
    #         for i3 in range(i2 + 1, len(qmap)):
    #             res.append(get_test_4_qubit(center, i1, i2, i3))

    # # five-qubit circuits
    # for i1 in range(1, len(qmap)):
    #     for i2 in range(i1 + 1, len(qmap)):
    #         for i3 in range(i2 + 1, len(qmap)):
    #             for i4 in range(i3 + 1, len(qmap)):                    
    #                 res.append(get_test_5_qubit(center, i1, i2, i3, i4))
    return res

def get_golden(testcases):
    golden = []
    for init_testcase in testcases:
        circ = QuantumCircuit(9, 9)
        circ = init_testcase(circ)
        # print(circ)
        sv = Statevector(circ)
        # print(sv)
        dm = DensityMatrix(sv)
        # print(dm)
        qubits_to_trace = [q for q in range(9) if q not in qmap]
        dm_partial = partial_trace(dm, qargs=qubits_to_trace)
        print(f"Размерность полной матрицы плотности: {dm.dim}")
        print(f"Размерность частичной матрицы плотности: {dm_partial.dim}")
        golden.append(dm_partial)
    return golden

def make_measures(backend: Runner, measures, testcases, num_shots=2048):
    freqs_list = []
    for init_testcase in testcases:
        freqs = measure(backend, measures, init_testcase, 9, num_shots)
        freqs_list.append(freqs)
    return freqs_list

# def trunc_freqs(freqs_list):
#     truncated = []
#     for freqs in freqs_list:
#         fr = []
#         print(freqs)
#         for idx in qmap:
#             fr.append[freqs[idx]]
#         truncated.append(fr)

num_qubits = len(qmap)
print("Starting QST test")
print(f"Number of qubits: {num_qubits}")
print("Prepairing testcases...")
testcases = get_init_states()
print(f"Prepaired {len(testcases)} testcases")
print("Calculating golden...")
golden = get_golden(testcases)
print("Prepairing measure ops...")
ops_ = silly_measure_operators(num_qubits)
print("Transpile measure ops...")
ops = transpile_measure_operators(ops_, 9, qmap)
backend = Runner("StateVector")
print("Make measures...")
freqs_list = make_measures(backend, ops, testcases, 2048)
# freqs_list = trunc_freqs(freqs_list)
results = np.asarray(freqs_list)
print(results.shape)
now = datetime.datetime.now()
formatted_now = now.strftime("%Y-%m-%d_%H-%M-%S")
np.save(f'{backend}_{formatted_now}', results)

print("Decompose ops by basis...")
basis = qiskit.quantum_info.pauli_basis(num_qubits).to_matrix()[1:]
basis = [b / norm(b) for b in basis]
projectors = [get_povm_op(*label) for label in ops_]
print(f"basis = {len(basis)}, {basis[0].shape}")
print(f"projectors = {len(projectors)}, {projectors[0].shape}")
X = decompose_by_basis(basis, projectors)
dim = 2 ** num_qubits

for k, freqs in enumerate(freqs_list):
    print(f"Testcase {k}:")
    
    print("Solve exact...")
    theta = solve_exact(X, freqs - 1./dim)
    rho = np.eye(dim) / dim + reconstruct_by_basis(basis, theta)
    # print(f'Trace = {np.trace(rho)}')
    eigenvalues = np.linalg.eigh(rho)[0]
    positive_defined = np.all(eigenvalues > 0)
    # print(f'Positive-defined = {positive_defined}')
    # print(f'Eigenvalues: {eigenvalues}')

    # print(f'Fidelity: {state_fidelity(expected, expected)}')
    # print(f'Purity: {real.purity()}')
    from matplotlib import pyplot as plt
    real = DensityMatrix(rho)
    diff = real - golden[k]
    print(f'Max diff: {np.max(np.abs(diff))}')
    print(f'L2 norm: {np.sqrt(np.sum(np.square(np.abs(diff))))}')
    # real.draw('city')
    # plt.show()
    diff.draw('city')
    plt.show()
    # X@theta = freqs
    print("Solve ls...")
    theta = solve_ls(X, freqs - 1./dim)
    rho = np.eye(dim) / dim + reconstruct_by_basis(basis, theta)
    real = DensityMatrix(rho)
    diff = real - golden[k]
    print(f'Max diff: {np.max(np.abs(diff))}')
    print(f'L2 norm: {np.sqrt(np.sum(np.square(np.abs(diff))))}')