import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import random_statevector, Statevector
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

def prepare_random_two_qubit_state():
    """Подготавливает случайное состояние для системы из двух кубитов"""
    # Создаем случайный вектор состояния для 2 кубитов
    random_state = random_statevector(4)  # 4 = 2^2 для 2 кубитов
    
    # Создаем квантовую схему
    qc = QuantumCircuit(2)
    
    # Подготавливаем случайное состояние используя Statevector
    qc.initialize(random_state, [0, 1])
    
    return qc, random_state

def measure_pauli_basis(qc, pauli_operator, shots=1000):
    """Измеряет состояние в заданном базисе Паули"""
    # Создаем копию схемы для измерений
    measure_circuit = qc.copy()
    measure_circuit.barrier()
    
    # Применяем соответствующие вращения для измерения в нужном базисе
    if pauli_operator == 'XX':
        # Для XX: ничего не делаем, измеряем в вычислительном базисе
        measure_circuit.measure_all()
    elif pauli_operator == 'YY':
        # Для YY: применяем S†-H к каждому кубиту
        measure_circuit.sdg(0)
        measure_circuit.sdg(1)
        measure_circuit.h(0)
        measure_circuit.h(1)
        measure_circuit.measure_all()
    elif pauli_operator == 'ZZ':
        # Для ZZ: применяем H к каждому кубиту
        measure_circuit.h(0)
        measure_circuit.h(1)
        measure_circuit.measure_all()
    elif pauli_operator == 'XY':
        # Для XY: на первом кубите ничего, на втором S†-H
        measure_circuit.sdg(1)
        measure_circuit.h(1)
        measure_circuit.measure_all()
    elif pauli_operator == 'XZ':
        # Для XZ: на первом кубите ничего, на втором H
        measure_circuit.h(1)
        measure_circuit.measure_all()
    elif pauli_operator == 'YX':
        # Для YX: на первом кубите S†-H, на втором ничего
        measure_circuit.sdg(0)
        measure_circuit.h(0)
        measure_circuit.measure_all()
    elif pauli_operator == 'YZ':
        # Для YZ: на первом кубите S†-H, на втором H
        measure_circuit.sdg(0)
        measure_circuit.h(0)
        measure_circuit.h(1)
        measure_circuit.measure_all()
    elif pauli_operator == 'ZX':
        # Для ZX: на первом кубите H, на втором ничего
        measure_circuit.h(0)
        measure_circuit.measure_all()
    elif pauli_operator == 'ZY':
        # Для ZY: на первом кубите H, на втором S†-H
        measure_circuit.h(0)
        measure_circuit.sdg(1)
        measure_circuit.h(1)
        measure_circuit.measure_all()
    
    # Запускаем симуляцию
    simulator = AerSimulator()
    compiled_circuit = transpile(measure_circuit, simulator)
    job = simulator.run(compiled_circuit, shots=shots)
    result = job.result()
    counts = result.get_counts()
    
    # Вычисляем ожидаемое значение
    expectation_value = calculate_expectation_value(counts, pauli_operator)
    
    return expectation_value

def calculate_expectation_value(counts, pauli_operator):
    """Вычисляет ожидаемое значение оператора Паули из результатов измерений"""
    total = sum(counts.values())
    expectation = 0
    
    for outcome, count in counts.items():
        # Преобразуем строку '00', '01' и т.д. в значения ±1
        bit1, bit2 = int(outcome[0]), int(outcome[1])
        
        # Вычисляем значение для данного исхода
        if pauli_operator == 'XX':
            value = (-1)**(bit1 + bit2)  # X⊗X
        elif pauli_operator == 'YY':
            value = (-1)**(bit1 + bit2)  # Y⊗Y
        elif pauli_operator == 'ZZ':
            value = (-1)**(bit1 + bit2)  # Z⊗Z
        elif pauli_operator == 'XY':
            value = (-1)**(bit1 + bit2)  # X⊗Y
        elif pauli_operator == 'XZ':
            value = (-1)**(bit1 + bit2)  # X⊗Z
        elif pauli_operator == 'YX':
            value = (-1)**(bit1 + bit2)  # Y⊗X
        elif pauli_operator == 'YZ':
            value = (-1)**(bit1 + bit2)  # Y⊗Z
        elif pauli_operator == 'ZX':
            value = (-1)**(bit1 + bit2)  # Z⊗X
        elif pauli_operator == 'ZY':
            value = (-1)**(bit1 + bit2)  # Z⊗Y
        
        expectation += value * count / total
    
    return expectation

def main():
    """Основная функция программы"""
    print("=== Томография случайного состояния двух кубитов ===\n")
    
    # 1. Подготавливаем случайное состояние
    qc, random_state = prepare_random_two_qubit_state()
    
    print("Случайное состояние подготовлено:")
    print(f"Вектор состояния: {random_state.data}")
    print(f"Схема глубины: {qc.depth()}")
    
    # 2. Измеряем во всех возможных базисах Паули
    pauli_operators = ['XX', 'XY', 'XZ', 'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ']
    measurement_vector = []
    
    print("\nИзмерения в базисах Паули:")
    print("-" * 40)
    
    for pauli_op in pauli_operators:
        expectation = measure_pauli_basis(qc, pauli_op, shots=5000)
        measurement_vector.append(expectation)
        print(f"<{pauli_op}> = {expectation:.4f}")
    
    # 3. Выводим итоговый вектор измерений
    print("\n" + "="*50)
    print("ИТОГОВЫЙ ВЕКТОР ИЗМЕРЕНИЙ:")
    print("="*50)
    
    measurement_array = np.array(measurement_vector)
    print(f"Вектор измерений: {measurement_array}")
    print(f"Размерность вектора: {measurement_array.shape}")
    print(f"Норма вектора: {np.linalg.norm(measurement_array):.4f}")
    
    # Визуализация
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(pauli_operators, measurement_vector)
    plt.title('Измерения в базисах Паули')
    plt.xlabel('Оператор Паули')
    plt.ylabel('Ожидаемое значение')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Визуализация действительной и мнимой частей состояния
    real_part = np.real(random_state.data)
    imag_part = np.imag(random_state.data)
    x = range(len(real_part))
    plt.bar(x, real_part, alpha=0.7, label='Действительная')
    plt.bar(x, imag_part, alpha=0.7, label='Мнимая')
    plt.title('Компоненты случайного состояния')
    plt.xlabel('Базисное состояние')
    plt.ylabel('Амплитуда')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return measurement_array

if __name__ == "__main__":
    result_vector = main()