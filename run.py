import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from quantumclient.client import QuantumClient
from quantumclient import qasm_to_cirucit

token = "3965d59b-3b1f-48e8-86ef-398a9dd71f40"
qclient = QuantumClient(token)
back_fmn = qclient.remote("Snowdrop 4Q ver2")

# вывод параметров квантового вычислителя
print(f"{back_fmn.num_qubits=}")
print(f"{back_fmn.basis_gates=}")
print(f"{back_fmn.coupling_map=}")
print(f"{back_fmn.f_q(1)=}")
print(f"{back_fmn.t1(1)=}")
print(f"{back_fmn.gate_error('rx', 1)=}")
print(f"{back_fmn.gate_error('cz', (2,1))=}")