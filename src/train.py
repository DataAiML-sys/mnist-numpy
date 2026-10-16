import numpy as np
import time 
from network import NeuralNetwork

print("Loading data...")
X_train = np.load("data/X_train.npy")
Y_train = np.load("data/Y_train.npy")
X_test = np.load("data/X_test.npy")
Y_test = np.load("data/Y_test.npy")

print(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}")
print(f"X_test:  {X_test.shape}, Y_test:  {Y_test.shape}")

epochs = 10
batch_size = 64
learning_rate = 0.1

net = NeuralNetwork()

print(f"\nTraining: {epochs} epochs, batch_size={batch_size}, lr={learning_rate}")

start_time = time.time()

for epoch in range(epochs):
    permutation = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[permutation]
    Y_train_shuffled = Y_train[permutation]
    num_batches = len(X_train) // batch_size
    total_loss = 0.0
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        X_batch = X_train_shuffled[start:end]
        Y_batch = Y_train_shuffled[start:end]
        probs = net.forward(X_batch)
        eps = 1e-10
        loss = -np.sum(Y_batch * np.log(probs + eps)) / batch_size
        total_loss += loss
        net.backward(X_batch, Y_batch)
        net.W1 -= learning_rate * net.dW1
        net.b1 -= learning_rate * net.db1
        net.W2 -= learning_rate * net.dW2
        net.b2 -= learning_rate * net.db2
    test_preds = net.predict(X_test)
    test_true = np.argmax(Y_test, axis=1)
    test_acc = np.mean(test_preds == test_true) * 100
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch+1}/{epochs} | loss={avg_loss:.4f} | test_accuracy={test_acc:.2f}%")
elapsed = time.time() - start_time
print(f"\nTraining took {elapsed:.1f} seconds")

train_preds = net.predict(X_train)
train_true = np.argmax(Y_train, axis=1)
train_acc = np.mean(train_preds == train_true) * 100

print(f"\nFinal accuracy:")
print(f"  Train: {train_acc:.2f}%")
print(f"  Test:  {test_acc:.2f}%")

np.savez("data/model_weights.npz", W1=net.W1, b1=net.b1, W2=net.W2, b2=net.b2)
print("\nWeights saved to data/model_weights.npz")