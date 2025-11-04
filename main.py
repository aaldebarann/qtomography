import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, random_density_matrix
from qiskit.quantum_info import Pauli
from qiskit.primitives import StatevectorEstimator as Estimator

def measure_observables(density_matrix, num_shots=1000):
    """
    Измеряет наблюдаемые X, Y, Z для системы из одного кубитов
    
    Args:
        density_matrix: матрица плотности системы 2x2
        num_shots: количество измерений для каждой наблюдаемой
    
    Returns:
        vector: трехмерный вектор [⟨X⟩, ⟨Y⟩, ⟨Z⟩]
    """
    
    # Создаем наблюдаемые (операторы Паули)
    X_obs = Pauli('X')
    Y_obs = Pauli('Y') 
    Z_obs = Pauli('Z')
    
    # Используем Estimator для вычисления средних значений
    estimator = Estimator()
    
    # Создаем квантовую схему (просто для интерфейса, не выполняем реальные измерения)
    qc = QuantumCircuit(1)
    
    # Вычисляем математические ожидания
    pub = (qc, [X_obs, Y_obs, Z_obs])
    job = estimator.run([pub])
    
    expectation_X = job.result()[0].values[0]
    expectation_Y = job.result()[0].values[1]
    expectation_Z = job.result()[0].values[2]
    
    return np.array([expectation_X, expectation_Y, expectation_Z])

def alternative_measurement_method(density_matrix, num_shots=1000):
    """
    Альтернативный метод измерения через симуляцию квантовых схем
    """
    results = []
    
    # Для каждой наблюдаемой создаем свою схему измерения
    observables = ['X', 'Y', 'Z']
    
    for obs in observables:
        # Создаем квантовую схему
        qc = QuantumCircuit(1, 1)
        
        # Подготавливаем состояние согласно матрице плотности
        # В реальном эксперименте это делалось бы томографией,
        # но для симуляции используем встроенные методы
        
        # Добавляем измерение в нужном базисе
        if obs == 'X':
            qc.h(0)  # Преобразование Адамара для измерения X
        elif obs == 'Y':
            qc.sdg(0)  # S† гейт для измерения Y
            qc.h(0)
        
        qc.measure(0, 0)
        
        # В реальном коде здесь был бы запуск на симуляторе или реальном устройстве
        # Для простоты вычисляем математическое ожидание аналитически
        
        if obs == 'X':
            expectation = np.real(np.trace(density_matrix @ np.array([[0, 1], [1, 0]])))
        elif obs == 'Y':
            expectation = np.real(np.trace(density_matrix @ np.array([[0, -1j], [1j, 0]])))
        else:  # 'Z'
            expectation = np.real(np.trace(density_matrix @ np.array([[1, 0], [0, -1]])))
        
        results.append(expectation)
    
    return np.array(results)

# Основная программа
if __name__ == "__main__":
    # Генерируем случайную матрицу плотности для одного кубита
    print("Генерация случайной матрицы плотности для системы из двух кубитов...")
    random_dm = random_density_matrix(2, seed=np.random.randint(1000))
    
    print("\nСгенерированная матрица плотности:")
    print(random_dm)
    
    # Проверяем, что матрица плотности корректна (след = 1, эрмитова)
    print(f"\nСлед матрицы: {np.trace(random_dm):.6f}")
    # print(f"Эрмитовость: {np.allclose(random_dm, random_dm.conjugate().T)}")
    
    # Измеряем наблюдаемые
    print("\nИзмерение наблюдаемых...")
    measurement_vector = measure_observables(random_dm, num_shots=1000)
    
    print("\nРезультат измерений (вектор [⟨X⟩, ⟨Y⟩, ⟨Z⟩]):")
    print(f"[{measurement_vector[0]:.6f}, {measurement_vector[1]:.6f}, {measurement_vector[2]:.6f}]")
    
    # Альтернативный метод для проверки
    print("\nАльтернативный метод измерения:")
    alt_vector = alternative_measurement_method(random_dm)
    print(f"[{alt_vector[0]:.6f}, {alt_vector[1]:.6f}, {alt_vector[2]:.6f}]")
    
    # Проверяем согласованность методов
    print(f"\nРазница между методами: {np.linalg.norm(measurement_vector - alt_vector):.6e}")
    
    # Дополнительная информация о состоянии
    print(f"\nДополнительная информация:")
    print(f"Норма вектора Блоха: {np.linalg.norm(measurement_vector):.6f}")
    print(f"Чистота состояния: {np.trace(random_dm @ random_dm):.6f}")