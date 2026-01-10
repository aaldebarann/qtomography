import numpy as np

def norm(matrix):
    return np.sqrt(np.trace(matrix.T.conj() @ matrix))

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

def state_sdgh(circ): # prepare 1/sqrt(2)( |0> + |1> ) state
    for i in range(circ.num_qubits):
        circ.sdg(i)
        circ.h(i)
    return circ

def state_Bell(circ): # prepare Bell state
    pass