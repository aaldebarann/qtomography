import numpy as np
from qiskit import QuantumCircuit
from runner import Runner

def povm(proj, outcome, num_qubits, init_state, num_shots):
    circuit = QuantumCircuit(num_qubits, num_qubits)
    circuit = init_state(circuit)

    for i, p in enumerate(proj):
        ii = num_qubits - i - 1
        if(p == 'X'):
            circuit.h(ii)
        elif(p == 'Y'):
            circuit.sdg(ii)
            circuit.h(ii)
        elif(p == 'Z'):
            pass
        elif(p == 'I'):
            pass
        else:
            raise ValueError(f"UNKNOW proj at index {i}: {proj}")
        circuit.measure([ii], [ii])
    
    sampler = Runner("StateVector")
    pub = circuit
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

def measure(backend: Runner, projectors: list, init_state, num_qubits, num_shots=20480):
    f = np.empty(shape=(len(projectors),), dtype=np.float32)
    batch = []
    proj, outcome = zip(*projectors)
    N = len(projectors)
    print(f"Run {N} measures")
    for k in range(N):
        circuit = QuantumCircuit(num_qubits, num_qubits)
        circuit = init_state(circuit)

        for i, p in enumerate(proj[k]):
            ii = num_qubits - i - 1
            if(p == 'X'):
                circuit.h(ii)
            elif(p == 'Y'):
                circuit.sdg(ii)
                circuit.h(ii)
            elif(p == 'Z'):
                pass
            elif(p == 'I'):
                pass
            else:
                raise ValueError(f"UNKNOW proj at index {i}: {proj[k]}")
            circuit.measure([ii], [ii])
        
        batch.append(circuit)
    
    job = backend.run(batch, shots=num_shots)

    # Extract the result for the 0th pub (this example only has one pub).
    result = job.result()

    if backend.backend_type == "StateVector":
        for k in range(N):
            counts = result[k].data.c.get_counts()
            # print(f"{proj}, {outcome}: {counts}")
            f[k] = 0
            if(outcome[k] in counts):
                f[k] = counts[outcome[k]] / num_shots
    else:
        for k in range(N):
            counts = result.get_counts()[k]
            f[k] = 0
            if(outcome[k] in counts):
                f[k] = counts[outcome[k]] / num_shots

    # for i, p in enumerate(projectors):
    #     proj, outcome = p
    #     f[i] = povm(proj, outcome, num_qubits, init_state, num_shots)
    return f

def get_povm_op(observable, outcome):
    vecs = {
        'Z': {'0': np.asarray([[1], [0]]),
              '1': np.asarray([[0], [1]])},
        'X': {'0': np.asarray([[1/np.sqrt(2)],[1/np.sqrt(2)]]),
              '1': np.asarray([[1/np.sqrt(2)], [-1/np.sqrt(2)]])},
        'Y': {'0': np.asarray([[1/np.sqrt(2)], [1j/np.sqrt(2)]]),
              '1': np.asarray([[1/np.sqrt(2)], [-1j/np.sqrt(2)]])}
    }
    operator = np.asarray(1)
    for obs, out in zip(observable, outcome):
        m = vecs[obs][out] @ vecs[obs][out].T.conj()
        operator = np.kron(operator, m)
    return operator

def another_silly_measure_operators(num_qubits):
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
        else:
            ops.pop()
        if len(ops) == dim2:
            break
    return labels

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
        else:
            ops.pop()
        if len(ops) == dim2:
            break
    return labels

def transpile_measure_operators(measures, num_total, measured_idx):
    transpiled = []
    for item in measures:
        obs, out = item
        tobs = ''
        tout = ''
        for i in range(num_total):
            if i in measured_idx:
                tobs += obs[measured_idx.index(i)]
                tout += out[measured_idx.index(i)]
            else:
                tobs += 'Z'
                tout += '0'
        transpiled.append((tobs, tout))
    return transpiled


def manual_measure_operators(num_qubits):
    if num_qubits == 1:
        return [
            ('X', '0'),
            ('Y', '0'),
            ('Z', '0')
        ]
    elif num_qubits == 2:
        return [
            ('XX', '00'),
            ('XX', '01'),
            ('XX', '11'),
            ('YY', '00'),
            ('YY', '01'),
            ('YY', '11'),
            ('ZZ', '00'),
            ('ZZ', '01'),
            ('ZZ', '11'),
            ('XY', '00'),
            ('XZ', '00'),
            ('YZ', '00'),
            ('YX', '00'),
            ('ZX', '00'),
            ('ZY', '00'),
        ]
    elif num_qubits == 3:
        return [
            ('XXX', '000'), ('XXX', '001'), ('XXX', '010'), ('XXX', '011'),
            ('XXX', '100'), ('XXX', '101'), ('XXX', '110'),
            ('YYY', '000'), ('YYY', '001'), ('YYY', '010'), ('YYY', '011'),
            ('YYY', '100'), ('YYY', '101'), ('YYY', '110'),
            ('ZZZ', '000'), ('ZZZ', '001'), ('ZZZ', '010'), ('ZZZ', '011'),
            ('ZZZ', '100'), ('ZZZ', '101'), ('ZZZ', '110'),
            ('XZZ', '000'),
            ('YZZ', '000'),
            ('XZZ', '010'),
            ('YZZ', '010'),
            ('ZXZ', '000'),
            ('ZXZ', '100'),
            ('XXZ', '000'),
            ('YXZ', '000'),
            ('ZYZ', '000'),
            ('ZYZ', '100'),
            ('XYZ', '000'),
            ('YYZ', '000'),
            ('XZZ', '001'),
            ('YZZ', '001'),
            ('ZXZ', '001'),
            ('YXZ', '001'),
            ('ZYZ', '001'),
            ('XYZ', '001'),
            ('ZZX', '000'),
            ('ZZX', '100'),
            ('XZX', '000'),
            ('YZX', '000'),
            ('ZZX', '010'),
            ('YZX', '010'),
            ('ZXX', '000'),
            ('YXX', '000'),
            ('ZYX', '000'),
            ('ZYX', '100'),
            ('XYX', '000'),
            ('YYX', '000'),
            ('ZZY', '000'),
            ('ZZY', '100'),
            ('XZY', '000'),
            ('YZY', '000'),
            ('ZZY', '010'),
            ('XZY', '010'),
            ('ZXY', '000'),
            ('ZXY', '100'),
            ('XXY', '000'),
            ('YXY', '000'),
            ('ZYY', '000'),
            ('XYY', '000'),
        ]
    else:
        pass

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
    x = np.linalg.inv(K) @ A.T.conj() @ b
    return x

