from qiskit_quantuminspire.qi_provider import QIProvider
from qiskit.primitives import StatevectorSampler

class Runner:
    def __init__(self, backend_type):
        self.backend_type=backend_type
        if backend_type == "StateVector":
            self.backend = StatevectorSampler()
        elif backend_type == "Tuna-9":
            provider = QIProvider()
            self.backend = provider.get_backend("Tuna-9")
        elif backend_type == "QX emulator":
            provider = QIProvider()
            self.backend = provider.get_backend("QX emulator")
        else:
            raise ValueError(f"Unknown backend \"{backend_type}\"!")

    def __str__(self):
        return self.backend_type

    def run(self, *args, **kwargs):
        return self.backend.run(*args, **kwargs)
        
    
    