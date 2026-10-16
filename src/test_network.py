import numpy as np
from network import NeuralNetwork

net = NeuralNetwork()

print("Shapes:")
print(f"  W1: {net.W1.shape}")
print(f"  b1: {net.b1.shape}")
print(f"  W2: {net.W2.shape}")
print(f"  b2: {net.b2.shape}")

X_test = np.random.rand(2, 784)
Y_test = np.zeros((2, 10))
Y_test[0, 3] = 1.0
Y_test[1, 7] = 1.0

probs = net.forward(X_test)
print(f"\nForward pass:")
print(f"  probs shape: {probs.shape}")
print(f"  probs[0]: {probs[0]}")
print(f"  sum probs[0]: {probs[0].sum():.4f}")

preds = net.predict(X_test)
print(f"\nPredictions: {preds}")
print("(random, because weights are random)")

net.backward(X_test, Y_test)
print(f"\nBackward pass - gradient shapes:")
print(f"  dW1: {net.dW1.shape}   (expected (784, 128))")
print(f"  db1: {net.db1.shape}   (expected (128,))")
print(f"  dW2: {net.dW2.shape}   (expected (128, 10))")
print(f"  db2: {net.db2.shape}   (expected (10,))")

print("\nAll good!")