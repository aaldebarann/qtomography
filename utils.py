import numpy as np
from qiskit import QuantumCircuit
import qiskit.quantum_info
from qiskit.quantum_info import DensityMatrix, random_density_matrix
from qiskit.quantum_info import Pauli
from qiskit.primitives import StatevectorEstimator
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector

def povm(proj, outcome, num_qubits, init_state, num_shots):
    circuit = QuantumCircuit(num_qubits, num_qubits)
    circuit = init_state(circuit)

    for i in range(num_qubits):
        if(proj[i] == 'X'):
            circuit.h(i)
        elif(proj[i] == 'Y'):
            circuit.sdg(i)
            circuit.h(i)
        elif(proj[i] == 'Z'):
            pass
        elif(proj[i] == 'I'):
            pass
        else:
            print(f"UNKNOW proj at index {i}: {proj}")
        circuit.measure([i], [i])
    sampler = StatevectorSampler()
    pub = (circuit)
    job = sampler.run([pub], shots=num_shots)

    # Extract the result for the 0th pub (this example only has one pub).
    result = job.result()[0]

    raw = result.data.c.array

    counts = result.data.c.get_counts()
    # print(f"{proj}, {outcome}: {counts}")
    freq = 0
    if(outcome in counts):
        freq = counts[outcome] / num_shots
    # print(f"freq = {freq}")
    return freq

def measure(projectors: list, init_state, num_qubits, num_shots=2048):
    f = np.empty(shape=(len(projectors),), dtype=np.complex128)
    for i, p in enumerate(projectors):
        proj, outcome = p
        f[i] = povm(proj, outcome, num_qubits, init_state, num_shots)
    return np.asarray(f).astype(float)

def state_I(circ): # prepare |0> state
    return circ
def state_X(circ): # prepare |1> state
    for i in range(circ.num_qubits):
        circ.x(i)
    return circ
def state_H(circ): # prepare 1/sqrt(2)( |0> + |1> ) state
    for i in range(circ.num_qubits):
        circ.h(i)
    return circ

def norm(matrix):
    return np.sqrt(np.trace(matrix.T.conj() @ matrix))


def not_so_silly_measure_operators(num_qubits):
    def is_independent(system):
        flatten = [matrix.flatten() for matrix in system]
        flatten_matrix = np.asarray(flatten)
        rank = np.linalg.matrix_rank(flatten_matrix)
        return len(system) == rank

    vecs = [
        {'vec': np.asarray([[1], [0]]), 'outcome': '0', 'label': 'Z'},  # |0>
        {'vec': np.asarray([[0], [1]]), 'outcome': '1', 'label': 'Z'},  # |1>
        {'vec': np.asarray([[1/np.sqrt(2)], [1/np.sqrt(2)]]), 'outcome': '0', 'label': 'X'},  # |+>
        {'vec': np.asarray([[1/np.sqrt(2)], [-1/np.sqrt(2)]]), 'outcome': '1', 'label': 'X'},  # |->
        {'vec': np.asarray([[1/np.sqrt(2)], [1j/np.sqrt(2)]]), 'outcome': '0', 'label': 'Y'},  # |+i>
        {'vec': np.asarray([[1/np.sqrt(2)], [-1j/np.sqrt(2)]]), 'outcome': '1', 'label': 'Y'},  # |-i>
    ]
    dim = 2 ** num_qubits # dim of state vector
    dim2 = dim ** 2 # dim of density matrix
    combinations = 6 ** num_qubits # possible combinations of pauli eigvectors

    ops = [np.eye(dim),]
    labels = []
    for j in range(3):
        for i in range(dim - 1):
            k = i
            m = vecs[2*j + k % 2]['vec'] @ vecs[2*j + k % 2]['vec'].T.conj()
            l = vecs[2*j + k % 2]['label']
            o = vecs[2*j + k % 2]['outcome']
            for _ in range(num_qubits - 1):
                k = k // 2
                m = np.kron(m, vecs[2*j + k % 2]['vec'] @ vecs[2*j + k % 2]['vec'].T.conj())
                l += vecs[2*j + k % 2]['label']
                o += vecs[2*j + k % 2]['outcome']
            ops.append(m)
            if is_independent(ops):
                labels.append((l, o))
                print(f"pick {l}, {o}")
            else:
                ops.pop()
            if len(ops) == dim2:
                break

    for i in range(combinations):
        k = i
        m = vecs[k % 6]['vec'] @ vecs[k % 6]['vec'].T.conj()
        l = vecs[k % 6]['label']
        o = vecs[k % 6]['outcome']
        for _ in range(num_qubits - 1):
            k = k // 6
            m = np.kron(m, vecs[k % 6]['vec'] @ vecs[k % 6]['vec'].T.conj())
            l += vecs[k % 6]['label']
            o += vecs[k % 6]['outcome']
        ops.append(m)
        if is_independent(ops):
            labels.append((l, o))
            print(f"pick {l}, {o}")
        else:
            ops.pop()
        if len(ops) == dim2:
            break
    return ops[1:], labels

def silly_measure_operators(num_qubits):
    def is_independent(system):
        flatten = [matrix.flatten() for matrix in system]
        flatten_matrix = np.asarray(flatten)
        rank = np.linalg.matrix_rank(flatten_matrix)
        return len(system) == rank

    vecs = [
        {'vec': np.asarray([[1], [0]]), 'outcome': '0', 'label': 'Z'},  # |0>
        {'vec': np.asarray([[0], [1]]), 'outcome': '1', 'label': 'Z'},  # |1>
        {'vec': np.asarray([[1/np.sqrt(2)], [1/np.sqrt(2)]]), 'outcome': '0', 'label': 'X'},  # |+>
        {'vec': np.asarray([[1/np.sqrt(2)], [-1/np.sqrt(2)]]), 'outcome': '1', 'label': 'X'},  # |->
        {'vec': np.asarray([[1/np.sqrt(2)], [1j/np.sqrt(2)]]), 'outcome': '0', 'label': 'Y'},  # |+i>
        {'vec': np.asarray([[1/np.sqrt(2)], [-1j/np.sqrt(2)]]), 'outcome': '1', 'label': 'Y'},  # |-i>
    ]
    dim = 2 ** num_qubits # dim of state vector
    dim2 = dim ** 2 # dim of density matrix
    combinations = 6 ** num_qubits # possible combinations of pauli eigvectors

    ops = [np.eye(dim),]
    labels = []
    for i in range(combinations):
        k = i
        m = vecs[k % 6]['vec'] @ vecs[k % 6]['vec'].T.conj()
        l = vecs[k % 6]['label']
        o = vecs[k % 6]['outcome']
        for _ in range(num_qubits - 1):
            k = k // 6
            m = np.kron(m, vecs[k % 6]['vec'] @ vecs[k % 6]['vec'].T.conj())
            l += vecs[k % 6]['label']
            o += vecs[k % 6]['outcome']
        ops.append(m)
        if is_independent(ops):
            labels.append((l, o))
            print(f"pick {l}, {o}")
        else:
            ops.pop()
        if len(ops) == dim2:
            break
    return ops[1:], labels
    


def manual_measure_operators(num_qubits):
    vecs = {
        '0': {'vec': np.asarray([[1], [0]]), 'outcome': '0', 'label': 'Z'},  # |0>
        '1': {'vec': np.asarray([[0], [1]]), 'outcome': '1', 'label': 'Z'},  # |1>
        '+': {'vec': np.asarray([[1/np.sqrt(2)], [1/np.sqrt(2)]]), 'outcome': '0', 'label': 'X'},  # |+>
        '-': {'vec': np.asarray([[1/np.sqrt(2)], [-1/np.sqrt(2)]]), 'outcome': '1', 'label': 'X'},  # |->
        '+i': {'vec': np.asarray([[1/np.sqrt(2)], [1j/np.sqrt(2)]]), 'outcome': '0', 'label': 'Y'},  # |+i>
        '-i': {'vec': np.asarray([[1/np.sqrt(2)], [-1j/np.sqrt(2)]]), 'outcome': '1', 'label': 'Y'},  # |-i>
    }
    oneop = [
            (vecs['+']['vec'] @ vecs['+']['vec'].T.conj()),
            (vecs['+i']['vec'] @ vecs['+i']['vec'].T.conj()),
            (vecs['0']['vec'] @ vecs['0']['vec'].T.conj())
    ]
    oneop_labels = [('X', '0'), ('Y', '0'), ('Z', '0')]
    if num_qubits == 1:
        return oneop, oneop_labels
    elif num_qubits == 2:
        oneop = [
            (vecs['+']['vec'] @ vecs['+']['vec'].T.conj()),
            (vecs['-']['vec'] @ vecs['-']['vec'].T.conj()),
            (vecs['+i']['vec'] @ vecs['+i']['vec'].T.conj()),
            (vecs['-i']['vec'] @ vecs['-i']['vec'].T.conj()),
            (vecs['0']['vec'] @ vecs['0']['vec'].T.conj()),
            (vecs['1']['vec'] @ vecs['1']['vec'].T.conj())
        ]
        twoop = [
            np.kron(oneop[0], oneop[0]),
            np.kron(oneop[1], oneop[1]),
            np.kron(oneop[0], oneop[1]),
            np.kron(oneop[2], oneop[2]),
            np.kron(oneop[3], oneop[3]),
            np.kron(oneop[2], oneop[3]),
            np.kron(oneop[4], oneop[4]),
            np.kron(oneop[5], oneop[5]),
            np.kron(oneop[4], oneop[5]),
            np.kron(oneop[0], oneop[2]),
            np.kron(oneop[0], oneop[4]),
            np.kron(oneop[2], oneop[4]),
            np.kron(oneop[2], oneop[0]),
            np.kron(oneop[4], oneop[0]),
            np.kron(oneop[4], oneop[2]),
        ]
        # twoop = [
        #     np.kron(oneop[4], oneop[4]),
        #     np.kron(oneop[5], oneop[4]),
        #     np.kron(oneop[0], oneop[4]),
        #     np.kron(oneop[2], oneop[4]),
        #     np.kron(oneop[4], oneop[5]),
        #     np.kron(oneop[0], oneop[5]),
        #     np.kron(oneop[2], oneop[5]),
        #     np.kron(oneop[4], oneop[0]),
        #     np.kron(oneop[5], oneop[0]),
        #     np.kron(oneop[0], oneop[0]),
        #     np.kron(oneop[2], oneop[0]),
        #     np.kron(oneop[4], oneop[2]),
        #     np.kron(oneop[5], oneop[2]),
        #     np.kron(oneop[0], oneop[2]),
        #     np.kron(oneop[2], oneop[2]),
        # ]
        twoop_labels = [('XX', '00'), ('XX', '11'), ('XX', '01'),
                        ('YY', '00'), ('YY', '11'), ('YY', '01'),
                        ('ZZ', '00'), ('ZZ', '11'), ('ZZ', '01'),
                        ('XY', '00'), ('XZ', '00'), ('YZ', '00'),
                        ('YX', '00'), ('ZX', '00'), ('ZY', '00')]
        # twoop_labels = [('ZZ', '00'), ('ZZ', '10'), ('XZ', '00'),
        #                 ('YZ', '00'), ('ZZ', '01'), ('XZ', '01'),
        #                 ('YZ', '01'), ('ZX', '00'), ('ZX', '10'),
        #                 ('XX', '00'), ('YX', '00'), ('ZY', '00'),
        #                 ('ZY', '10'), ('XY', '00'), ('YY', '00')]
        return twoop, twoop_labels
    else:
        pass

def decompose_by_basis(basis: list, matrices: list): # basis must be orthonormal
    shape = (len(matrices), len(basis))
    A = np.zeros(shape=shape, dtype=np.complex128)
    for i, m in enumerate(matrices):
        for j, b in enumerate(basis):
            A[i][j] = np.trace(b.T.conj() @ m)
    return A

def reconstruct_by_basis(basis: list, coefs: list):
    r = np.zeros(shape=basis[0].shape, dtype=np.complex128)
    for b, coef in zip(basis, coefs):
        r += coef * b
    return r

# solving Ax = b problem

def solve_exact(A, b):
    # exact solution
    # print(f'eigenvals(A) = {np.linalg.eigh(A)[0]}')
    A_inv = np.linalg.inv(A)
    x = A_inv @ b
    return x

def solve_ls(A, b):
    # least squares solution
    K = A.T.conj() @ A
    # print(f'eigenvals(K) = {np.linalg.eigh(K)[0]}')
    x = np.linalg.inv(K) @ A.t.conj() @ b
    return x

# def mnk(f):
#     A = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]) / (np.sqrt(2))
#     A_inv = np.linalg.inv(A)

#     theta = np.linalg.inv(A.T @ A) @ A.T @ (f - 0.5)
#     # print("theta = ", theta)
#     # Reconstruct density matrix
#     I_norm = np.eye(2)/2
#     X_norm = np.array([[0, 1], [1, 0]])/np.sqrt(2)
#     Y_norm = np.array([[0, -1j], [1j, 0]])/np.sqrt(2)
#     Z_norm = np.array([[1, 0], [0, -1]])/np.sqrt(2)

#     rho = I_norm + theta[0]*X_norm + theta[1]*Y_norm + theta[2]*Z_norm
#     return rho
