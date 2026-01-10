from qiskit_quantuminspire.qi_provider import QIProvider

provider = QIProvider()

for item in provider.backends():
    print(item)



backend_name = "Tuna-9"
backend = provider.get_backend(name=backend_name)

# from matplotlib.pyplot import imshow
# imshow(backend.coupling_map.draw())
# backend.coupling_map.draw().show




