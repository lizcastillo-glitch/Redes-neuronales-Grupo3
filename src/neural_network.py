"""
Implementación desde cero de una red neuronal feedforward usando NumPy.
"""
class NeuralNetwork:
    def __init__(self, layers, activation='relu', init='xavier'):
        ...
    
    def train(self, X, y, epochs, lr, batch_size=32, verbose=True):
        ...
    
    def predict(self, X):
        ...
